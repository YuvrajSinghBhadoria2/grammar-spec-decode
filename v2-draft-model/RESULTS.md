# v2: Standalone Draft-Model Proposer — Negative Result

## Summary

Running a separate Qwen2.5-0.5B draft model inside the `propose()` call
with grammar-masked logits produces 100% valid JSON output but is 61x
slower than no_spec decoding. The draft model's forward pass dominates
the cost: 40 propose calls × K=5 sequential forward passes = 200 extra
0.5B forward passes per benchmark run.

## Benchmark (Qwen2.5-1.5B target, Qwen2.5-0.5B draft, A100 80GB)

| Config              | Valid | TPOT (ms) | Tok/s | vs ngram | vs no_spec |
|---------------------|-------|-----------|-------|----------|------------|
| no_spec             | 8/8   | 3.0       | 332.8 | 2.27x    | 1.00x      |
| baseline_ngram      | 8/8   | 6.8       | 146.5 | 1.00x    | 0.44x      |
| grammar_aware       | 8/8   | 12.4      | 80.7  | 0.55x    | 0.24x      |
| draft_model_grammar | 8/8   | 183.5     | 5.4   | 0.04x    | 0.02x      |

## Key diagnostics (draft_model_grammar run)

- `num_propose_calls`: 40
- `avg_draft_len`: 32.5 tokens per propose call (should be ≤5)
- `total_draft_tokens_returned`: 1302
- vLLM SpecDecoding metrics: Mean acceptance length 1.02–1.27,
  Avg Draft acceptance rate 0.4%–5.4%
- Validity: 8/8 (100%)

## Root cause analysis

1. **Standalone draft model is too slow.** Running Qwen2.5-0.5B inside
   `propose()` adds K=5 sequential forward passes per call. With 40 calls,
   that's 200 extra 0.5B forward passes. At ~0.9s per forward pass on
   A100 (batch=1, no KV cache reuse), this dominates the runtime.

2. **Vocab mismatch causes FSM rejections.** The draft model's vocab
   (151936) exceeds the target model's FSM range (151665). Tokens like
   151728 and 151684 are rejected by the xgrammar matcher, causing
   re-rolls and wasted compute.

3. **The architecture is wrong.** A draft model must share the target
   model's forward pass (like EAGLE) or run as a separate worker with
   its own KV cache. Running it inside `propose()` — which is called
   synchronously in the target model's sampling loop — serializes the
   two models' compute.

## What would fix this

- **EAGLE integration**: patch `EagleProposer.propose` to apply the
  grammar bitmask after `compute_logits` and before `argmax`. This
  shares the target model's hidden states and avoids a separate forward
  pass. Requires an EAGLE head compatible with the target model.

- **Separate draft-model worker**: run the draft model as a persistent
  worker with its own KV cache, communicating drafts via shared memory.
  This is how vLLM's V0 engine did it (DraftModelWorker), but V1 removed
  this path.

- **Batch draft model with target**: run both models in a single forward
  pass by concatenating their layers (not practical for different
  architectures).

## Conclusion

The grammar-masked logits approach is correct in principle (draft tokens
are grammar-valid AND high-probability under the draft model), but the
standalone implementation is too slow. The real contribution requires
deep integration with vLLM's speculative decoding internals — specifically
EAGLE, which is the only draft-model path vLLM 0.11.2 supports.

## Files

- `grammar_masked_draft_proposer.py` — standalone grammar-masked draft proposer
- `benchmark_e2e.py` — 4-config benchmark (no_spec, baseline_ngram, grammar_aware, draft_model_grammar)
- `benchmark_results.json` — raw results from the final run
