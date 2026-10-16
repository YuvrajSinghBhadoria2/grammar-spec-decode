# v0: Grammar-Aware Spec Decoding with Truncation

## Result

**100% valid JSON outputs** with grammar-aware filtering. But naive truncation
is 2x slower than baseline ngram spec decode.

## Benchmark (A100 80GB, Qwen2.5-1.5B bf16, 8 prompts)

| Config | Valid | TPOT (ms) | Tok/s | Speedup |
|--------|-------|-----------|-------|---------|
| no_spec | 8/8 | 3.3 | 302.8 | 2.01x |
| baseline_ngram | 8/8 | 6.7 | 150.3 | 1.00x |
| grammar_aware (truncate) | 8/8 | 13.5 | 74.3 | 0.49x |

## Grammar filter stats
- Drafts proposed: 344
- Drafts filtered: 292
- Filter rate: 84.9%
- Grammar active requests: 8

## Analysis

The filter rate of 84.9% means almost all ngram-proposed drafts are grammar-
invalid. With truncation, the draft length collapses to near-zero tokens,
eliminating the speculation benefit while still paying the FSM overhead.

**Root cause**: ngram spec decode proposes drafts by copying prior tokens
from the prompt. These tokens are natural language, not JSON-schema-valid.
When constrained by a JSON grammar, nearly all ngram drafts are rejected.

**Next step**: Replace filtered tokens with grammar-valid alternatives from
the bitmask (the `replace` strategy), not truncate. This preserves draft
length and turns guaranteed rejections into valid (potentially accepted)
drafts.

## Files
- `grammar_aware_proposer.py` — GrammarAwareProposer with truncate strategy
- `benchmark_e2e.py` — 3-config benchmark (no_spec, baseline_ngram, grammar_aware)
- `test_grammar_proposer.py` — 4 unit tests (all passing)
- `benchmark_results.json` — raw results from A100
