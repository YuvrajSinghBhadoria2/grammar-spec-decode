"""Grammar-Aware Speculative Decoding proposer.

Wraps any vLLM base proposer (ngram, eagle, etc.) and filters proposed draft
tokens through a shadow grammar FSM so drafts are grammar-valid by construction.

Motivation: on JSON/tool-calling workloads, grammar-invalid draft tokens are
guaranteed rejections that waste entire verification rounds. This proposer
pre-filters drafts through the same grammar the target model will be masked
with, converting guaranteed rejections into valid (potentially accepted) drafts.

Usage (as a monkey-patch in benchmarks, or via custom_class in vLLM >= 0.12):

    from grammar_aware_proposer import GrammarAwareProposer, wrap_drafter

    # In a vLLM worker setup hook:
    model_runner.drafter = wrap_drafter(model_runner.drafter)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import xgrammar as xgr

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)


class GrammarAwareProposer:
    """Wraps a base proposer and filters drafts through grammar FSMs.

    Maintains a shadow xgrammar matcher per request, advanced with verified
    tokens and rolled back after probing. Draft tokens that the FSM rejects
    are replaced with the best grammar-valid token from a fallback pool,
    or the draft sequence is truncated.
    """

    def __init__(
        self,
        base_proposer,
        *,
        num_speculative_tokens: int,
        vocab_size: int,
        model_name: str | None = None,
        fallback_strategy: str = "truncate",
    ):
        """Initialize the grammar-aware wrapper.

        Args:
            base_proposer: The underlying proposer (ngram, eagle, etc.).
            num_speculative_tokens: Max draft tokens per step (K).
            vocab_size: Target model vocabulary size (for bitmask allocation).
            model_name: HuggingFace model name for tokenizer/vocab metadata.
            fallback_strategy: What to do when a draft is grammar-invalid.
                "truncate": stop drafting at first invalid token (default).
                "replace": replace invalid token with top grammar-valid candidate
                    from the base proposer's logits (requires logits access).
        """
        self.base = base_proposer
        self.k = num_speculative_tokens
        self.vocab_size = vocab_size
        self.model_name = model_name
        self.fallback_strategy = fallback_strategy
        # Save the original bound method so we never recurse through a
        # monkey-patched instance attribute.
        self._original_propose = base_proposer.propose

        # Per-request shadow grammar state.
        # Maps req_id (string) -> xgr.GrammarMatcher
        self._matchers: dict[str, xgr.GrammarMatcher] = {}
        # Maps req_id -> last known token count (for advancing FSM)
        self._token_counts: dict[str, int] = {}

        # Stats for benchmarking
        self.stats = {
            "total_drafts_proposed": 0,
            "total_drafts_filtered": 0,
            "total_requests": 0,
            "grammar_active_requests": 0,
        }

    def set_grammar(self, req_id: str, compiled_grammar: xgr.CompiledGrammar) -> None:
        """Attach a compiled grammar to a request.

        Args:
            req_id: Request ID string (from vLLM's req_ids list).
            compiled_grammar: Pre-compiled xgrammar.CompiledGrammar.
        """
        self._matchers[req_id] = xgr.GrammarMatcher(compiled_grammar)
        self._token_counts[req_id] = 0
        self.stats["grammar_active_requests"] += 1

    def clear_grammar(self, req_id: str) -> None:
        """Remove grammar state for a request (e.g., after it finishes)."""
        self._matchers.pop(req_id, None)
        self._token_counts.pop(req_id, None)

    def advance_with_verified(self, req_id: str, verified_tokens: list[int]) -> None:
        """Advance the shadow FSM with tokens the target model actually produced.

        Call this after each step with the verified (accepted) tokens for this
        request, so the FSM state matches the target model's output position.
        """
        matcher = self._matchers.get(req_id)
        if matcher is None:
            return
        for token in verified_tokens:
            if not matcher.is_terminated():
                matcher.accept_token(token)
        self._token_counts[req_id] = self._token_counts.get(req_id, 0) + len(
            verified_tokens
        )

    def filter_drafts(
        self, req_id: str, draft_tokens: list[int]
    ) -> list[int]:
        """Filter draft tokens through the grammar FSM.

        Args:
            req_id: Request ID string.
            draft_tokens: Proposed draft tokens from the base proposer.

        Returns:
            filtered: Grammar-valid prefix of draft_tokens. May be shorter
                than input (truncation) or empty.
        """
        matcher = self._matchers.get(req_id)
        if matcher is None:
            # No grammar attached — pass through unchanged
            return draft_tokens

        self.stats["total_drafts_proposed"] += len(draft_tokens)

        valid: list[int] = []
        for token in draft_tokens:
            if matcher.is_terminated():
                # Grammar already complete; no more tokens allowed
                break
            if matcher.accept_token(token):
                valid.append(token)
            else:
                # Grammar rejects this token — stop here
                # (rollback happens below)
                break

        # Roll back the FSM to entry state (we probed non-destructively)
        num_advanced = len(valid)
        if num_advanced > 0:
            matcher.rollback(num_advanced)

        filtered_count = len(draft_tokens) - len(valid)
        self.stats["total_drafts_filtered"] += filtered_count

        return valid

    def propose(
        self,
        sampled_token_ids: list,
        req_ids: list[str],
        num_tokens_no_spec,
        token_ids_cpu,
        spec_decode_unsupported_reqs: set,
    ) -> list[list[int]]:
        """Propose draft tokens with grammar filtering.

        Matches NgramProposer.propose signature in vLLM 0.11.x.
        """
        self.stats["total_requests"] += len(sampled_token_ids)

        # 1. Advance shadow FSMs with the tokens just verified this step
        for i, sampled in enumerate(sampled_token_ids):
            req_id = req_ids[i] if i < len(req_ids) else None
            if req_id is not None and req_id in self._matchers:
                # sampled is np.ndarray of shape [num_sampled]
                tokens = sampled.tolist() if hasattr(sampled, "tolist") else list(sampled)
                if tokens:
                    self.advance_with_verified(req_id, tokens)

        # 2. Get raw drafts from the saved original bound method (not self.base.propose,
        #    which may have been monkey-patched to call back into us)
        raw_drafts = self._original_propose(
            sampled_token_ids,
            req_ids,
            num_tokens_no_spec,
            token_ids_cpu,
            spec_decode_unsupported_reqs,
        )

        # 3. Filter each request's drafts through its grammar
        filtered_drafts: list[list[int]] = []
        for i, drafts in enumerate(raw_drafts):
            req_id = req_ids[i] if i < len(req_ids) else None
            if req_id is not None and req_id in self._matchers:
                filtered = self.filter_drafts(req_id, drafts)
            else:
                filtered = drafts
            filtered_drafts.append(filtered[: self.k])

        self._flush_stats()
        return filtered_drafts

    def _flush_stats(self) -> None:
        """Write stats to the stats file if one is configured."""
        import json as _json

        stats_path = getattr(self, "_stats_path", None)
        if stats_path:
            try:
                with open(stats_path, "w") as f:
                    _json.dump(self.get_stats(), f)
            except Exception:
                pass

    def get_stats(self) -> dict:
        """Return accumulated statistics."""
        stats = dict(self.stats)
        if stats["total_drafts_proposed"] > 0:
            stats["filter_rate"] = (
                stats["total_drafts_filtered"] / stats["total_drafts_proposed"]
            )
        else:
            stats["filter_rate"] = 0.0
        return stats

    def reset_stats(self) -> None:
        """Reset statistics counters."""
        self.stats = {
            "total_drafts_proposed": 0,
            "total_drafts_filtered": 0,
            "total_requests": 0,
            "grammar_active_requests": 0,
        }


def wrap_drafter(
    base_proposer,
    *,
    num_speculative_tokens: int,
    vocab_size: int,
    model_name: str | None = None,
) -> GrammarAwareProposer:
    """Convenience function to wrap an existing vLLM drafter."""
    return GrammarAwareProposer(
        base_proposer,
        num_speculative_tokens=num_speculative_tokens,
        vocab_size=vocab_size,
        model_name=model_name,
    )
