"""Grammar-masked draft-model proposer for vLLM 0.11.x speculative decoding.

Runs a small draft model (e.g. Qwen2.5-0.5B), applies an xgrammar bitmask to
the draft logits before argmax, and returns grammar-valid draft tokens.

This is the correct architecture for grammar-aware speculative decoding:
instead of post-hoc substitution (v1, diagnosed negative), the grammar
constrains the draft model's sampling distribution directly, so drafted
tokens are both grammar-valid AND high-probability under the draft model.

Interface matches NgramProposer.propose so it can be monkey-patched in
the same way as GrammarAwareProposer.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)


class GrammarMaskedDraftProposer:
    """Draft-model proposer with grammar-masked logits.

    Loads a small draft model, runs K sequential forward passes to produce
    draft tokens, and applies an xgrammar bitmask to the draft logits before
    argmax at each step. This ensures drafted tokens are grammar-valid
    without post-hoc substitution.
    """

    def __init__(
        self,
        draft_model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        *,
        num_speculative_tokens: int = 5,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        """Initialize the grammar-masked draft proposer.

        Args:
            draft_model_name: HuggingFace model name for the draft model.
            num_speculative_tokens: Max draft tokens per step (K).
            device: CUDA device for the draft model.
            dtype: Model dtype (bf16 recommended).
        """
        self.k = num_speculative_tokens
        self.device = device
        self.draft_model_name = draft_model_name

        # Load draft model + tokenizer
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("Loading draft model: %s", draft_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(draft_model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            draft_model_name,
            torch_dtype=dtype,
            device_map=device,
        )
        self.model.eval()

        # Per-request state
        # Maps req_id -> list of token IDs generated so far (draft context)
        self._contexts: dict[str, list[int]] = {}
        # Maps req_id -> xgr.GrammarMatcher (shadow FSM)
        self._matchers: dict = {}
        # Maps req_id -> compiled grammar (for matcher recreation)
        self._grammars: dict = {}
        # Maps req_id -> xgr.TokenizerInfo (for bitmask allocation)
        self._tokenizer_infos: dict = {}
        # Maps req_id -> bitmask cache (reused across steps)
        self._bitmasks: dict[str, torch.Tensor] = {}
        # Maps req_id -> vocab size from tokenizer_info (may differ from model vocab)
        self._vocab_sizes: dict[str, int] = {}

        self.reset_stats()
        self._stats_path: str | None = None

    def set_grammar(
        self,
        req_id: str,
        compiled_grammar,
        tokenizer_info=None,
    ) -> None:
        """Attach a grammar to a request.

        Args:
            req_id: Request ID string.
            compiled_grammar: xgr.CompiledGrammar for this request.
            tokenizer_info: xgr.TokenizerInfo (needed for bitmask allocation).
                If None, uses the draft model's tokenizer.
        """
        import xgrammar as xgr

        self._grammars[req_id] = compiled_grammar

        if tokenizer_info is not None:
            self._tokenizer_infos[req_id] = tokenizer_info
            vocab_size = tokenizer_info.vocab_size
        else:
            # Build TokenizerInfo from draft model tokenizer
            info = xgr.TokenizerInfo.from_huggingface(self.tokenizer)
            self._tokenizer_infos[req_id] = info
            vocab_size = info.vocab_size

        self._vocab_sizes[req_id] = vocab_size

        # Create matcher with rollback capacity for K tokens
        matcher = xgr.GrammarMatcher(
            compiled_grammar,
            max_rollback_tokens=self.k,
        )
        self._matchers[req_id] = matcher

        # Allocate bitmask (reused across steps)
        bitmask = xgr.allocate_token_bitmask(1, vocab_size)
        self._bitmasks[req_id] = bitmask

        # Initialize context with empty history
        self._contexts[req_id] = []

        self.stats["grammar_active_requests"] = len(self._matchers)
        logger.info(
            "Grammar attached to %s (vocab=%d)", req_id, vocab_size
        )

    def remove_grammar(self, req_id: str) -> None:
        """Detach grammar and clean up state for a request."""
        self._matchers.pop(req_id, None)
        self._grammars.pop(req_id, None)
        self._tokenizer_infos.pop(req_id, None)
        self._bitmasks.pop(req_id, None)
        self._vocab_sizes.pop(req_id, None)
        self._contexts.pop(req_id, None)
        self.stats["grammar_active_requests"] = len(self._matchers)

    def _fill_bitmask(self, req_id: str) -> bool:
        """Fill the bitmask for the current FSM state. Returns True on success."""
        matcher = self._matchers.get(req_id)
        bitmask = self._bitmasks.get(req_id)
        if matcher is None or bitmask is None or matcher.is_terminated():
            return False

        try:
            matcher.fill_next_token_bitmask(bitmask, 0)
            return True
        except Exception as e:
            logger.warning("Bitmask fill failed for %s: %s", req_id, e)
            return False

    def _apply_grammar_mask(
        self, req_id: str, logits: torch.Tensor
    ) -> torch.Tensor:
        """Apply the grammar bitmask to draft logits in-place.

        Args:
            req_id: Request ID.
            logits: Draft model logits, shape (1, vocab_size).

        Returns:
            Masked logits (same tensor, modified in-place).
        """
        import xgrammar as xgr

        if req_id not in self._matchers:
            return logits

        matcher = self._matchers[req_id]
        if matcher.is_terminated():
            # Grammar complete — no constraint
            return logits

        if not self._fill_bitmask(req_id):
            return logits

        bitmask = self._bitmasks[req_id]
        vocab_size = self._vocab_sizes.get(req_id)

        # apply_token_bitmask_inplace expects logits on same device as bitmask.
        # Bitmask is on CPU, logits on GPU. Move bitmask to GPU.
        bitmask_gpu = bitmask.to(logits.device)

        try:
            xgr.apply_token_bitmask_inplace(
                logits,
                bitmask_gpu,
                vocab_size=vocab_size,
            )
        except Exception as e:
            logger.warning("apply_token_bitmask_inplace failed for %s: %s", req_id, e)

        return logits

    def _draft_single_request(
        self, req_id: str, context_tokens: list[int]
    ) -> list[int]:
        """Generate K grammar-valid draft tokens for a single request.

        Runs K sequential forward passes of the draft model, applying the
        grammar bitmask to logits before argmax at each step.

        Args:
            req_id: Request ID.
            context_tokens: Full token history (prompt + generated so far).

        Returns:
            List of K draft token IDs (may be shorter if grammar terminates).
        """
        matcher = self._matchers.get(req_id)
        if matcher is None or matcher.is_terminated():
            return []

        draft_tokens: list[int] = []

        # Build input from context
        input_ids = torch.tensor(
            [context_tokens], dtype=torch.long, device=self.device
        )

        for step in range(self.k):
            if matcher.is_terminated():
                break

            # Forward pass
            with torch.no_grad():
                outputs = self.model(input_ids)
                logits = outputs.logits[:, -1, :]  # (1, vocab)

            # Apply grammar mask
            logits = self._apply_grammar_mask(req_id, logits)

            # Argmax (greedy, matches vLLM's draft selection)
            next_token = torch.argmax(logits, dim=-1)  # (1,)
            token_id = next_token.item()

            # Advance shadow FSM
            accepted = matcher.accept_token(token_id)
            if not accepted:
                # Should not happen after masking, but break defensively
                logger.warning(
                    "Draft token %d rejected by FSM for %s at step %d",
                    token_id, req_id, step,
                )
                break

            draft_tokens.append(token_id)

            # Append to input for next step
            input_ids = torch.cat(
                [input_ids, next_token.unsqueeze(0)], dim=1
            )

        # Roll back FSM to entry state (we probed non-destructively)
        num_advanced = len(draft_tokens)
        if num_advanced > 0:
            matcher.rollback(num_advanced)

        return draft_tokens

    def propose(
        self,
        sampled_token_ids: list,
        req_ids: list[str],
        num_tokens_no_spec,
        token_ids_cpu,
        spec_decode_unsupported_reqs: set,
    ) -> list[list[int]]:
        """Propose draft tokens with grammar-masked draft model.

        Matches NgramProposer.propose signature in vLLM 0.11.x.
        """
        self.stats["total_requests"] += len(sampled_token_ids)
        self.stats["num_propose_calls"] = (
            self.stats.get("num_propose_calls", 0) + 1
        )

        # 1. Advance shadow FSMs with the tokens just verified this step
        for i, sampled in enumerate(sampled_token_ids):
            req_id = req_ids[i] if i < len(req_ids) else None
            if req_id is not None and req_id in self._matchers:
                tokens = (
                    sampled.tolist() if hasattr(sampled, "tolist") else list(sampled)
                )
                if tokens:
                    # Advance FSM with verified tokens
                    matcher = self._matchers[req_id]
                    for t in tokens:
                        if not matcher.is_terminated():
                            matcher.accept_token(t)
                    # Update context
                    if req_id in self._contexts:
                        self._contexts[req_id].extend(tokens)

        # 2. Generate grammar-valid drafts for each request
        result_drafts: list[list[int]] = []
        for i, req_id in enumerate(req_ids):
            if req_id is not None and req_id in self._matchers:
                context = self._contexts.get(req_id, [])
                # Use the full token history from token_ids_cpu if available
                # (more reliable than maintaining our own context)
                if token_ids_cpu is not None and i < len(token_ids_cpu):
                    cpu_tokens = token_ids_cpu[i]
                    if hasattr(cpu_tokens, "tolist"):
                        context = cpu_tokens.tolist()
                    else:
                        context = list(cpu_tokens)

                draft = self._draft_single_request(req_id, context)
                result_drafts.append(draft[: self.k])

                self.stats["total_draft_tokens_returned"] = (
                    self.stats.get("total_draft_tokens_returned", 0)
                    + len(draft[: self.k])
                )
            else:
                # No grammar — return empty (fallback to no spec for this req)
                result_drafts.append([])

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
            "total_requests": 0,
            "grammar_active_requests": 0,
            "num_propose_calls": 0,
            "total_draft_tokens_returned": 0,
        }
