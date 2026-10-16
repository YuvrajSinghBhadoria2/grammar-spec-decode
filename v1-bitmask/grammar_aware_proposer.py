"""Grammar-Aware Speculative Decoding proposer.

Wraps any vLLM base proposer (ngram, eagle, etc.) and generates grammar-valid
draft tokens by construction using a shadow grammar FSM.

Motivation: on JSON/tool-calling workloads, grammar-invalid draft tokens are
guaranteed rejections that waste entire verification rounds. Rather than
filtering (which collapses draft length), this proposer uses the grammar FSM
to *generate* valid drafts: when the base proposer's token is invalid, the
FSM's bitmask identifies the valid alternative with the highest base-proposer
likelihood.

Core insight: grammar as draft *source*, not draft *filter*.

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
    """Wraps a base proposer and generates grammar-valid drafts via bitmask.

    Maintains a shadow xgrammar matcher per request, advanced with verified
    tokens. When drafting, the FSM's bitmask restricts the base proposer's
    candidates to grammar-valid tokens only, preserving full draft length.
    """

    def __init__(
        self,
        base_proposer,
        *,
        num_speculative_tokens: int,
        vocab_size: int,
        model_name: str | None = None,
        fallback_strategy: str = "bitmask",
    ):
        """Initialize the grammar-aware wrapper.

        Args:
            base_proposer: The underlying proposer (ngram, eagle, etc.).
            num_speculative_tokens: Max draft tokens per step (K).
            vocab_size: Target model vocabulary size (for bitmask allocation).
            model_name: HuggingFace model name for tokenizer/vocab metadata.
            fallback_strategy: How to handle grammar-invalid draft tokens.
                "bitmask": substitute with the highest-likelihood grammar-valid
                    token from the bitmask (default, preserves draft length).
                "truncate": stop drafting at first invalid token (v0 behavior).
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
        # Per-request bitmask (reused across steps to avoid reallocation)
        self._bitmasks: dict[str, "np.ndarray"] = {}

        # Stats for benchmarking
        self.stats = {
            "total_drafts_proposed": 0,
            "total_drafts_filtered": 0,
            "total_drafts_substituted": 0,
            "total_requests": 0,
            "grammar_active_requests": 0,
        }

    def set_grammar(
        self,
        req_id: str,
        compiled_grammar: xgr.CompiledGrammar,
        tokenizer_info: "xgr.TokenizerInfo | None" = None,
    ) -> None:
        """Attach a compiled grammar to a request.

        Args:
            req_id: Request ID string (from vLLM's req_ids list).
            compiled_grammar: Pre-compiled xgrammar.CompiledGrammar.
            tokenizer_info: Optional xgr.TokenizerInfo for correct bitmask sizing.
                If None, uses self.vocab_size (may mismatch xgrammar's buffer).
        """
        self._matchers[req_id] = xgr.GrammarMatcher(
            compiled_grammar, max_rollback_tokens=self.k
        )
        self._token_counts[req_id] = 0
        # Allocate bitmask with the correct buffer size
        if tokenizer_info is not None:
            vocab = tokenizer_info.vocab_size
        else:
            vocab = self.vocab_size
        self._bitmasks[req_id] = xgr.allocate_token_bitmask(
            batch_size=1, vocab_size=vocab
        )
        self.stats["grammar_active_requests"] += 1

    def clear_grammar(self, req_id: str) -> None:
        """Remove grammar state for a request (e.g., after it finishes)."""
        self._matchers.pop(req_id, None)
        self._token_counts.pop(req_id, None)
        self._bitmasks.pop(req_id, None)

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
        # Invalidate bitmask cache — FSM state changed
        self._bitmask_cache_key = None

    def _fill_bitmask(self, req_id: str) -> bool:
        """Fill the bitmask for the current FSM state. Returns True on success.

        Does NOT scan the full bitmask — that's 11ms. Just fills it.
        """
        import sys as _sys

        matcher = self._matchers.get(req_id)
        if matcher is None or matcher.is_terminated():
            return False

        bitmask = self._bitmasks.get(req_id)
        if bitmask is None:
            return False

        # Check if already filled for this state
        cache_key = (req_id, self._token_counts.get(req_id, 0), matcher.is_terminated())
        if getattr(self, "_bitmask_cache_key", None) == cache_key:
            return True

        try:
            matcher.fill_next_token_bitmask(bitmask, 0)
            self._bitmask_cache_key = cache_key
            return True
        except Exception as e:
            print(f"  [BITMASK] fill failed: {e}", file=_sys.stderr)
            _sys.stderr.flush()
            return False

    def _is_token_valid(self, req_id: str, token_id: int) -> bool:
        """Check if a single token is grammar-valid at the current FSM state.

        O(1) bit check — no full scan.
        """
        if not self._fill_bitmask(req_id):
            return False

        bitmask = self._bitmasks.get(req_id)
        if bitmask is None:
            return False

        import torch

        if isinstance(bitmask, torch.Tensor):
            word_idx = token_id // 32
            bit_pos = token_id % 32
            row = bitmask[0]
            if word_idx >= row.shape[0]:
                return False
            return bool(int(row[word_idx].item()) & (1 << bit_pos))
        return False

    def _get_first_valid_token(self, req_id: str) -> int | None:
        """Find the first valid token in the bitmask without full scan.

        Scans words until it finds a non-zero word, then finds the first set bit.
        """
        if not self._fill_bitmask(req_id):
            return None

        bitmask = self._bitmasks.get(req_id)
        if bitmask is None:
            return None

        import torch

        if isinstance(bitmask, torch.Tensor):
            row = bitmask[0]
            for wi in range(row.shape[0]):
                w = int(row[wi].item())
                if w:
                    for b in range(32):
                        if w & (1 << b):
                            tid = wi * 32 + b
                            if tid < self.vocab_size:
                                return tid
        return None

    def _get_valid_token_set(self, req_id: str) -> set[int]:
        """Get the set of grammar-valid tokens at the current FSM state.

        Uses the bitmask to enumerate valid tokens. Returns empty set if
        matcher is terminated or no grammar attached.
        """
        matcher = self._matchers.get(req_id)
        if matcher is None or matcher.is_terminated():
            return set()

        bitmask = self._bitmasks.get(req_id)
        if bitmask is None:
            return set()

        # Fill bitmask with valid tokens at current state
        try:
            matcher.fill_next_token_bitmask(bitmask, 0)
        except Exception as e:
            import sys as _sys
            print(f"  [BITMASK] fill_next_token_bitmask FAILED: {e}", file=_sys.stderr)
            _sys.stderr.flush()
            return set()

        # Enumerate set bits in the bitmask
        # Bitmask is a torch.Tensor of int32, shape (batch, ceil(vocab/32))
        import torch

        if isinstance(bitmask, torch.Tensor):
            row = bitmask[0]  # first (only) batch element
            valid_tokens = set()
            for word_idx in range(row.shape[0]):
                word = int(row[word_idx].item())
                if word == 0:
                    continue
                for bit in range(32):
                    if word & (1 << bit):
                        token_id = word_idx * 32 + bit
                        if token_id < self.vocab_size:
                            valid_tokens.add(token_id)
            return valid_tokens
        return set()

    def _substitute_token(
        self, req_id: str, proposed_token: int, base_draft: list[int], position: int
    ) -> int:
        """Substitute an invalid token with the best grammar-valid alternative.

        Uses O(1) bit checks — no full bitmask scan.
        """
        # If proposed token is valid, use it
        if self._is_token_valid(req_id, proposed_token):
            return proposed_token

        # Heuristic: try common structural tokens first (fast path)
        structural_candidates = [198, 220, 25, 29890, 197]  # } , : " {
        for token in structural_candidates:
            if token != proposed_token and self._is_token_valid(req_id, token):
                return token

        # Fallback: find first valid token (early-exit scan)
        first_valid = self._get_first_valid_token(req_id)
        if first_valid is not None:
            return first_valid

        return proposed_token

    def draft_with_grammar(
        self, req_id: str, base_draft: list[int]
    ) -> tuple[list[int], int]:
        """Generate a grammar-valid draft by substituting invalid tokens.

        Preserves full draft length. For each position, if the base proposer's
        token is grammar-invalid, substitute with a valid alternative from
        the bitmask.

        Args:
            req_id: Request ID string.
            base_draft: Draft tokens from the base proposer.

        Returns:
            (valid_draft, num_substitutions): Grammar-valid draft and count
                of tokens that were substituted.
        """
        matcher = self._matchers.get(req_id)
        if matcher is None:
            # No grammar — pass through unchanged
            return base_draft, 0

        self.stats["total_drafts_proposed"] += len(base_draft)

        valid_draft: list[int] = []
        num_substituted = 0

        for position, proposed_token in enumerate(base_draft):
            if matcher.is_terminated():
                # Grammar complete; no more tokens allowed
                break

            # Try the proposed token first (cheap: 0.3µs)
            if matcher.accept_token(proposed_token):
                valid_draft.append(proposed_token)
                # FSM advanced — but don't invalidate cache yet.
                # We'll invalidate lazily when _fill_bitmask is next called.
                self._bitmask_cache_key = None
            else:
                # Roll back the failed accept_token call
                # (accept_token may have advanced state even on failure in some impls)
                # Re-check: if accept_token returned False, state should be unchanged
                # Substitute with a grammar-valid alternative
                substitute = self._substitute_token(
                    req_id, proposed_token, base_draft, position
                )

                # Try to accept the substitute
                if matcher.accept_token(substitute):
                    valid_draft.append(substitute)
                    num_substituted += 1
                    # FSM advanced — invalidate bitmask cache for next position
                    self._bitmask_cache_key = None
                else:
                    # Substitute also failed — truncate here
                    break

        # Roll back the FSM to entry state (we probed non-destructively)
        num_advanced = len(valid_draft)
        if num_advanced > 0:
            matcher.rollback(num_advanced)

        self.stats["total_drafts_substituted"] += num_substituted
        filtered_count = len(base_draft) - len(valid_draft)
        self.stats["total_drafts_filtered"] += filtered_count

        return valid_draft, num_substituted

    def filter_drafts(
        self, req_id: str, draft_tokens: list[int]
    ) -> list[int]:
        """Filter draft tokens through the grammar FSM (v0 truncate strategy).

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
        """Propose draft tokens with grammar-aware generation.

        Matches NgramProposer.propose signature in vLLM 0.11.x.
        """
        self.stats["total_requests"] += len(sampled_token_ids)
        self.stats["num_propose_calls"] = self.stats.get("num_propose_calls", 0) + 1

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

        # 3. Generate grammar-valid drafts for each request
        result_drafts: list[list[int]] = []
        for i, drafts in enumerate(raw_drafts):
            req_id = req_ids[i] if i < len(req_ids) else None
            if req_id is not None and req_id in self._matchers:
                if self.fallback_strategy == "bitmask":
                    valid, _sub = self.draft_with_grammar(req_id, drafts)
                else:
                    valid = self.filter_drafts(req_id, drafts)
            else:
                valid = drafts
            truncated = valid[: self.k]
            result_drafts.append(truncated)
            # Track returned draft length for acceptance-rate diagnostics
            if req_id is not None and req_id in self._matchers:
                self.stats["total_draft_tokens_returned"] = (
                    self.stats.get("total_draft_tokens_returned", 0) + len(truncated)
                )

        self._flush_stats()
        return result_drafts

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
            stats["substitution_rate"] = (
                stats["total_drafts_substituted"] / stats["total_drafts_proposed"]
            )
        else:
            stats["filter_rate"] = 0.0
            stats["substitution_rate"] = 0.0
        # Derived acceptance-rate diagnostics
        num_calls = stats.get("num_propose_calls", 0)
        returned = stats.get("total_draft_tokens_returned", 0)
        if num_calls > 0:
            stats["avg_draft_len"] = returned / num_calls
        else:
            stats["avg_draft_len"] = 0.0
        return stats

    def reset_stats(self) -> None:
        """Reset statistics counters."""
        self.stats = {
            "total_drafts_proposed": 0,
            "total_drafts_filtered": 0,
            "total_drafts_substituted": 0,
            "total_requests": 0,
            "grammar_active_requests": 0,
            "num_propose_calls": 0,
            "total_draft_tokens_returned": 0,
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
        fallback_strategy="bitmask",
    )
