# Draft: vLLM issue / design proposal

> Not yet posted. For Yuvraj to review, adapt voice, and submit.
> Target: vllm-project/vllm — design-proposal or bug-adjacent issue,
> referencing the open roadmap item on structured outputs + spec decode.
>
> Pre-post checklist:
> 1. Decide whether to attach a minimal xgrammar `rollback()` repro
>    (T2/T3 micro-tests were session-transcript only — not archived), or
>    split observation 2 into a separate issue on the xgrammar repo.
> 2. Label the validity tables as single-session measurements (12 prompts
>    weather / 8 prompts tool schema) — n is small.
> 3. Fill final repo URL (currently the public grammar-spec-decode repo).

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
3. Do **not** rely on `GrammarMatcher.rollback()` for the
   drafted-but-rejected window: in our measurements it desyncs after a
   permanent accepted prefix when the window contains forced-continuation
   tokens (see observation 2). Instead, build a **fresh matcher per propose
   call**, replay the committed token history (`sampled_token_ids` from
   `prepare_next_token_ids_*`), mask, then discard the matcher at propose
   exit. Measured cost is acceptable; FSM replay rejections 0/2387.
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
instead. The advantage holds at every draft length tested: K=3 1.15×,
K=5 1.19×, K=7 1.17× (best absolute: K=3 grammar, 1.25× vs no_spec).

On a second, enum-heavy tool-call schema (nested enums + arrays):

| Config | Valid JSON | TPOT (ms) | tok/propose |
|---|---|---|---|
| no_spec | 5/8 | 11.02 | — |
| EAGLE3 (unmasked) | **3/8** | 13.54 | 1.203 |
| **EAGLE3 + grammar mask** | **5/8** | **10.37 (1.31× vs baseline)** | **1.807 (+50.2%)** |

Two further observations relevant to this roadmap item:

1. **Constrained spec decode currently degrades validity** on hard schemas
   (3/8 vs 5/8 no_spec — the unmasked drafts push decoding into
   degeneration loops that hit the token cap). Grammar-masking the drafts
   *restores* no_spec validity (5/8) while also being faster.
2. **Avoid `GrammarMatcher.rollback()` for this use case (xgrammar 0.1.25)**:
   in our measurements it desyncs after a permanent accepted prefix when the
   rolled-back window contains forced-continuation tokens (e.g. char-splatted
   enum values) — the matcher returns without error but stops reflecting the
   true grammar state (minimal repro available on request). A correct
   implementation should snapshot/restore matcher state or create a fresh
   matcher per propose and replay committed history (the prototype's v4 mode;
   FSM replay rejections 0/2387, zero rollback calls).

## Where this would live

`vllm/v1/spec_decode/eagle.py` — `EagleProposer.propose()` currently calls
`self.model.compute_logits(...)` then `argmax`. A `ConstrainedProposer`
wrapper (or a bitmask hook on the proposer) could mirror how the target-side
guided-sampling path integrates xgrammar, reusing the existing per-engine
`XGrammarBackend`/`TokenizerInfo`.

A working prototype (monkey-patched `EagleProposer`, flag-gated, fresh
matcher per propose + committed-history replay, zero rollback calls) is
benchmarked here: https://github.com/YuvrajSinghBhadoria2/grammar-spec-decode

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

Prototype + benchmarks (vLLM 0.11.2, xgrammar 0.1.25):
https://github.com/YuvrajSinghBhadoria2/grammar-spec-decode
