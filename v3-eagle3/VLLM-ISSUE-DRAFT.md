# Draft: vLLM issue / design proposal

> Not yet posted. For Yuvraj to review, adapt voice, and submit.
> Target: vllm-project/vllm — design-proposal or bug-adjacent issue,
> referencing the open roadmap item on structured outputs + spec decode.

---

**Title:** [Spec Decode] Compose grammar-constrained sampling with EAGLE speculative decoding

**Body:**

## Problem

Speculative decoding and grammar-constrained sampling (`structured_outputs`,
guided decoding) do not compose in vLLM today. Under structured outputs the
target verifier only accepts grammar-valid tokens, so *unconstrained* draft
models waste most of their draft budget on tokens the FSM will reject:

- Measured with vLLM 0.11.2 + xgrammar + Qwen2.5-7B-Instruct (A100, batch=1,
  agentic JSON workload, K=5): unmasked EAGLE3 accepted **1.44** draft tokens
  per propose call and was **0.97× slower than no speculation**.
- In an earlier n-gram draft experiment on the same workload, **84.9%** of
  drafted tokens were rejected by the grammar FSM (292/344).

This matches the open roadmap item:
*“Improve robustness with tool calling + reasoning and structured outputs”*
(SIG Spec Decode, Q3 2026 roadmap tracking issue #48168).

## Proposal: grammar-masked draft logits

Apply the grammar bitmask **inside the draft loop**, after `compute_logits`
and before `argmax`, at every speculative position — not only at final
verification:

1. `Eagle3LlamaForCausalLM.compute_logits` already scatters reduced
   draft-vocab logits into a **target-vocab-sized** tensor via `d2t`
   (with `-inf` elsewhere), so the standard xgrammar bitmask
   (`apply_token_bitmask_inplace`) applies directly with no vocab mapping.
2. Maintain one `xgrammar.GrammarMatcher` per request. At propose position
   `t >= 1`, advance the matcher with the token drafted at `t-1`
   (`self.input_ids[:batch]` in `EagleProposer.propose`); fill the bitmask
   for each active request row; mask; then `argmax`.
3. On propose exit, `rollback()` the number of drafted tokens the matcher
   accepted; the target's actual accept/reject decisions are reconciled at
   the next propose call by replaying committed tokens
   (`sampled_token_ids` from `prepare_next_token_ids_*`).
4. If the EAGLE3 draft-vocab subset contains no grammar-valid token for a
   row, fall back to the first grammar-valid token (worst case: one wasted
   draft slot; validity unaffected — measured fallback rate: 0/4486 masks).

Measured effect (same hardware/workload as above; 12 distinct agentic
prompts, 3 timing iterations, rotated config order, diagnostics disabled
in the hot path):

| Config | TPOT (ms) | tok/propose | vs no_spec | vs baseline |
|---|---|---|---|---|
| no_spec | 11.09 ± 0.01 | — | 1.00× | — |
| EAGLE3 (unmasked) | 11.47 ± 0.04 | 1.443 | 0.967× | 1.00× |
| **EAGLE3 + grammar mask** | **9.61 ± 0.03** | **1.969 (+36.5%)** | **1.154×** | **1.193×** |

Acceptance rises because **39.1%** of draft-position argmaxes change under
the mask (1756/4486 positions) — the mask removes grammar-invalid
high-probability drafts so the next-most-likely *valid* token is drafted
instead.

## Where this would live

`vllm/v1/spec_decode/eagle.py` — `EagleProposer.propose()` currently calls
`self.model.compute_logits(...)` then `argmax`. A `ConstrainedProposer`
wrapper (or a bitmask hook on the proposer) could mirror how the target-side
guided-sampling path integrates xgrammar, reusing the existing per-engine
`XGrammarBackend`/`TokenizerInfo`.

A working prototype (monkey-patched `EagleProposer`, flag-gated, with
per-request matchers + rollback) is benchmarked here:
[link to repo — fill in after publication]

## Known limitations (prototype)

- xgrammar matcher runs on CPU: one small host sync per draft position;
  a GPU-resident FSM would remove it.
- JSON grammars permit degeneration loops (e.g., endless legal whitespace
  tokens) — a general constrained-decoding issue, not spec-decode-specific.
- Evaluated at batch=1 only; batched propose + bitmask indices path
  (`apply_token_bitmask_inplace(..., indices=...)`) is ready but untested.
- One model/head pair (Qwen2.5-7B + community EAGLE3 head) and two JSON
  schemas so far.

## Repro

Prototype + benchmarks (vLLM 0.11.2, xgrammar 0.1.25): [repo link]
