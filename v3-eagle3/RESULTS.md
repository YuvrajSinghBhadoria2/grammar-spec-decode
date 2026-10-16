# v3: Grammar-Masked EAGLE3 Speculative Decoding — Positive Result

## Summary

Applying an xgrammar JSON-schema bitmask to EAGLE3 draft logits inside
vLLM 0.11.2's `EagleProposer.propose()` (batch=1, agentic JSON workload)
improves draft acceptance by **+36.5%** and end-to-end TPOT by **19.3%**
versus unmasked EAGLE3. Grammar masking turns EAGLE3 from a net-**loss**
on this workload (0.97x vs no_spec) into a net-**win** (1.15x vs
no_spec).

## Headline results (A100 80GB, Qwen2.5-7B-Instruct target,
`ruipeterpan/Qwen2.5-7B-Instruct_EAGLE3_UltraChat` head, K=5,
max_num_seqs=1, 12 distinct agentic prompts, 3 iterations, temp=0)

| Config | Valid | TPOT ms (mean±std) | tok/propose | vs no_spec | vs baseline |
|---|---|---|---|---|---|
| no_spec | 33/36 | 11.09 ± 0.01 | — | 1.00x | — |
| baseline EAGLE3 (unmasked) | 33/36 | 11.47 ± 0.04 | 1.443 ± 0.000 | 0.967x | 1.00x |
| **EAGLE3 + grammar mask** | 33/36 | **9.61 ± 0.03** | **1.969 ± 0.000** | **1.154x** | **1.193x** |

- Acceptance proxy `tokens/propose` ratio grammar/baseline = **1.365**
  (+36.5%); grammar needs 917 propose calls vs 1250 for the same
  ~1800-token workload (−27%).
- Per-iteration TPOT, rotated config order (no order effect):
  - no_spec: 11.09 / 11.10 / 11.08
  - baseline: 11.49 / 11.50 / 11.41
  - grammar: 9.65 / 9.58 / 9.60
- Frozen decision rule (grammar beats baseline TPOT in every timing
  iteration AND acceptance ratio > 1.0): **met in 3/3 iterations**.

## Methodology (best practice)

- Timing runs contain **zero diagnostic GPU syncs** in the hot path;
  headroom instrumentation lives in a separate gated DIAG run
  (`od_diag` flag). The DIAG run measured TPOT 10.01 ms vs 9.61 ms
  clean — i.e. the diagnostics cost ~4% and would have contaminated the
  comparison.
- **12 distinct prompts** (temp=0 made the earlier 12 copies of one
  prompt non-independent).
- **3 timing iterations with rotated config order** to cancel
  thermal/cache order effects; reported mean±std.
- Warmup request (32 tokens) absorbs first-request cost; the one-time
  grammar build (1.84 s: tokenizer load 1.3 s + TokenizerInfo 0.5 s)
  happens lazily inside the worker during warmup, excluded from TPOT.
- Identical prompts/schema across all three configs within each
  iteration; TPOT = wall time / decode tokens (prefill included, equal
  across configs).
- `enforce_eager=True` for spec configs (vLLM 0.11.2 EAGLE CUDA-graph
  `pad_for_cudagraph` IndexError); no_spec uses cudagraphs, matching
  each path's supported configuration.
- `propose_calls` includes ~19 warmup propose calls (≤2% denominator
  bias, near-equal across the two spec configs; does not affect the
  conclusion).

## Headroom diagnostic (separate DIAG run, 12 prompts, 917 propose calls)

- `mask_applied/mask_rows`: 4486/4486 — every draft position masked.
- `mask_changed` (argmax actually altered by the mask):
  **1756/4486 = 39.1%** of draft positions. Substantial real headroom.
- `fallback_first_valid`: 0 — the EAGLE3 draft-vocab subset (16000 via
  d2t) always contains at least one grammar-valid token here; the
  fallback path never fired.
- FSM tracking: `accepted_committed` 1823, `rejected_committed` 15,
  `terminated` 99, `draft_advances` 3591 — committed-token replay stays
  consistent with grammar state (15 rejects = stop-token edge cases).

## Validity (33/36 all configs — explained, not confounded)

A per-prompt probe (no spec, temp=0) found prompt 3 (Sydney wedding)
the sole failure, identical under all configs: the model emits the
complete tool call, then enters a degenerate `" \n\n "` whitespace loop
until the 512-token cap (`finish=length`), so the JSON never closes.
Whitespace tokens are *grammatically legal*, so neither target-side
constrained sampling nor draft-side masking prevents this — grammar
constrains structure, not repetition degeneration. Equal across
configs, so comparisons are unaffected. Every other prompt: valid.

## Key implementation facts (vLLM 0.11.2)

- Only ngram/suffix/medusa/eagle/eagle3/mtp proposers exist;
  `draft_model` raises NotImplementedError; no `custom_class` hook.
  The patch therefore monkey-patches `EagleProposer` at class level,
  gated at propose-time by a flag file on shared FS (engine workers are
  spawned processes that do not re-run `__main__`, but do receive
  parent-side class patches).
- `Eagle3LlamaForCausalLM.compute_logits` scatters reduced draft-vocab
  logits into a **target-vocab-sized** tensor (d2t) with -inf elsewhere
  → the xgrammar bitmask applies directly after `compute_logits`,
  before argmax, at each of the K draft positions.
- The wrapper advances the per-request shadow FSM with the previous
  drafted token (`self.input_ids[:batch]`) at positions ≥1, replays
  committed tokens from `prepare_next_token_ids_{padded,cpu}` at propose
  start, and rolls back drafted tokens if the target rejects them.
- Hot path per masked position: 1 small D2H sync (FSM accept, inherent
  to CPU-side grammar) + bitmask fill/apply (~50 µs). No argmax
  duplication, no `.item()` diagnostics (those were the earlier
  contamination).

## Why baseline EAGLE3 is net-negative here but masking wins

On these 12 diverse agentic JSON prompts, unmasked EAGLE3 accepts only
1.44 draft tokens per propose call — not enough to amortize K=5 draft
steps (0.967x vs no_spec). The mask raises acceptance to 1.97 while
adding ~0.5 ms/propose of FSM+bitmask cost, netting 1.19x over
baseline and 1.15x over no_spec. The mask both increases acceptance
(39.1% of argmaxes change) and reduces wasted draft work (27% fewer
propose calls for equal output).

## Caveats and limits

- Single workload class (weather-tool JSON schema), batch=1, one GPU
  (A100), one target/head pair, temp=0.
- Iterations replicate timing, not output distribution (deterministic
  outputs: identical token counts 1793/1804/1806 per iteration).
- `tokens/propose` is an end-to-end acceptance proxy derived from
  propose-call counts (1250 → 917 for equal output). vLLM 0.11.2's own
  SpecDecoding footers (present in `eagle3_grammar_b1.log`) corroborate
  the direction: Avg Draft acceptance rate 27–32% masked vs 20–21%
  unmasked, with acceptance shifted toward position 0 (0.59–0.63 vs
  0.39 per-position).
- xgrammar runs on CPU; a GPU-resident matcher could remove the
  remaining per-position D2H sync.
- Results have not been independently re-verified by a separate owner.

## Files

- `eagle3_grammar_b1.py` — benchmark + class-level grammar patch
  (flag-file gating: `od_enable`, `od_diag`)
- `eagle3_grammar_b1_results.json` — raw 3-iteration results
- `eagle3_grammar_diag_results.json` — headroom diagnostic run
- `probe_validity.py` / `probe_validity.log` — per-prompt validity
  probe identifying the prompt-3 whitespace degeneration
- `eagle3_grammar_b1.log` — full benchmark log (RESULTS footer at end)
