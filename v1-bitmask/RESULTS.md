# v1: Bitmask Substitution — Negative Result

## Summary

Bitmask substitution (replacing grammar-invalid ngram draft tokens with the
first valid token from the xgrammar bitmask) produces 100% valid JSON output
but is 2.3x slower than baseline ngram speculative decoding, and baseline
ngram is already 2x slower than no-spec decoding for this workload.

## Benchmark (Qwen2.5-1.5B-Instruct, A100 80GB, 8 weather prompts, K=5)

| Config            | Valid | TPOT (ms) | Tok/s  | Speedup vs no_spec |
|-------------------|-------|-----------|--------|--------------------|
| no_spec           | 8/8   | 3.1       | 323.3  | 1.00x              |
| baseline_ngram    | 8/8   | 6.0       | 165.7  | 0.51x              |
| grammar_aware     | 8/8   | 13.6      | 73.7   | 0.23x              |

## Key diagnostics (grammar_aware run)

- `num_propose_calls`: 40
- `avg_draft_len`: 8.7 tokens per propose call
- `substitution_rate`: 83.1% (290 of 349 draft tokens substituted)
- `filter_rate`: 0.0% (no drafts truncated)
- Validity: 8/8 (100%)

## Root cause analysis

1. **Spec decode is already a net loss here.** Baseline ngram (6.0ms TPOT)
   is 2x slower than no_spec (3.1ms TPOT). The 8 short weather prompts are
   too small for ngram speculation to amortize the verification cost.

2. **Substitution makes verification worse, not better.** Grammar-aware
   drafts are the same length as baseline (avg 8.7 tokens), but 83% of
   tokens are substituted with structural punctuation (`}`, `,`, `:`, `"`,
   `{`) that the target model would not have produced. The target model
   rejects these during verification, forcing more forward passes per
   accepted token.

3. **The structural heuristic is off-distribution.** The bitmask tells us
   which tokens are *grammatically valid*, but not which token the target
   model would *actually generate*. Substituting the first valid token
   (e.g., `}`) when the model wants a string character (`N` for "New York")
   guarantees rejection.

## What would fix this

The substitution needs to pick tokens that are both grammar-valid AND
high-probability under the target model. This requires either:

- **A draft model with grammar-masked logits**: run a small draft model,
  apply the bitmask to its logits, then sample. This is the correct
  architecture (EAGLE + grammar constraint).

- **Target-model logits in the proposer**: the current ngram proposer has
  no access to the target model's probability distribution. Without it,
  any substitution strategy is guessing.

## Files

- `grammar_aware_proposer.py` — bitmask substitution implementation
- `benchmark_e2e.py` — 3-config benchmark (no_spec, baseline_ngram, grammar_aware)
- `benchmark_results.json` — raw results from the final run
