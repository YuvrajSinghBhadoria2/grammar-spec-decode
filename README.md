# Grammar-Aware Speculative Decoding

**Making speculative decoding work under JSON-schema constraints — with honest, reproducible benchmarks.**

> **Status:** active research. Three instrumented iterations (v0–v2) are complete.
> All reach **100% valid JSON**; none has yet beaten the no-speculation baseline on speed.
> Negative results and root-cause analyses are preserved on purpose — see [Results](#results-at-a-glance).

## The problem

Speculative decoding drafts tokens by copying patterns from earlier tokens.
Under **grammar-constrained decoding** (JSON schemas, tool calling, structured outputs),
those drafts are usually *grammatically invalid* — so the grammar FSM rejects them,
acceptance collapses, and speculation stops paying off.

Measured in v0: under a JSON weather schema, **84.9% of n-gram drafts were rejected
by the grammar** (292 of 344 drafts). Speculative decoding is effectively broken for
exactly the workloads agents run most: tool calls and structured responses.

This is not a niche concern — [vLLM's Q3 2026 roadmap](https://github.com/vllm-project/vllm/issues/48168)
lists an open, unchecked item:

> *Improve robustness with tool calling + reasoning and structured outputs.* — SIG Spec Decode

## Approach

Draft **with** the grammar instead of against it: every strategy in this repo keeps
generation 100% schema-valid by construction, then measures what that costs in speed.

Three strategies were implemented, benchmarked end-to-end, and root-caused:

## Results at a glance

Benchmark: Qwen2.5-1.5B-Instruct (bf16) target on **NVIDIA A100**, vLLM 0.11.2,
xgrammar FSM, K=5 draft tokens, 8 JSON-schema-constrained prompts. Lower TPOT is better.

| Iteration | Strategy | Valid JSON | Grammar-aware TPOT | vs no-spec baseline | Verdict |
|-----------|----------|:----------:|-------------------:|--------------------:|---------|
| [v0](v0/) | Truncate invalid drafts | **8/8** | 13.5 ms (74.3 tok/s) | 0.24× (3.3 ms) | ❌ draft length collapses |
| [v1](v1-bitmask/) | Replace invalid drafts via bitmask | **8/8** | 13.6 ms (73.7 tok/s) | 0.23× (3.1 ms) | ❌ valid ≠ model-likely |
| [v2](v2-draft-model/) | Standalone draft model + masked logits | **8/8** | 183.5 ms (5.4 tok/s) | 0.02× (3.0 ms) | ❌ wrong architecture |

Raw numbers per run: each directory's `benchmark_results.json` and `RESULTS.md`.

## Key findings

1. **The problem is real and large.** 84.9% of n-gram drafts are grammar-invalid
   under a JSON schema (v0) — natural-language history simply doesn't contain
   schema-valid continuations.
2. **Grammar-valid does not mean model-likely.** Substituting the "first valid token"
   injected structural punctuation the target model would never produce; 83.1% of
   drafts were substituted and verification rejected them (v1).
3. **A draft model must share the target's forward pass.** Running a standalone
   0.5B drafter inside `propose()` serialized both models' compute: 61× slower than
   no-spec (v2). The viable path is EAGLE-style integration — apply the grammar
   bitmask to logits *after* the shared forward pass, before sampling.
4. **Baseline caveat:** n-gram speculation itself was ~2× *slower* than no-spec on
   these short prompts — speculation needs long outputs to amortize verification.
   Re-baselining on realistic long-output agentic traces is the next experiment.

## Repository layout

```
grammar-spec-decode/
├── README.md                    ← you are here
├── v0/                          ← strategy 1: truncate invalid drafts
│   ├── grammar_aware_proposer.py
│   ├── benchmark_e2e.py         ← 3-config end-to-end benchmark
│   ├── test_grammar_proposer.py ← 4 unit tests (CPU-only)
│   ├── RESULTS.md               ← numbers + root-cause analysis
│   └── benchmark_results.json   ← raw run output
├── v1-bitmask/                  ← strategy 2: bitmask substitution
└── v2-draft-model/              ← strategy 3: standalone grammar-masked drafter
```

## Reproducing

```bash
pip install "vllm==0.11.2" xgrammar numpy torch

# CPU-only unit tests (4 tests, no GPU needed)
python v0/test_grammar_proposer.py

# End-to-end benchmarks (NVIDIA GPU required)
python v0/benchmark_e2e.py
python v1-bitmask/benchmark_e2e.py
python v2-draft-model/benchmark_e2e.py
```

Each benchmark compares three configurations — `no_spec`, `baseline_ngram`,
`grammar_aware` (v2 adds `draft_model_grammar`) — and writes `benchmark_results.json`
next to the script.

## Next steps

1. **Workload realism:** re-baseline `no_spec` vs `baseline_ngram` on long-output
   agentic traces (500–2000-token JSON tool calls, repeated system prompts) —
   speculation must have room to amortize before grammar-awareness can matter.
2. **EAGLE integration:** apply the xgrammar bitmask after `compute_logits`,
   before sampling in the shared forward pass (the architecture v2's analysis
   points to).
3. **Frozen success criteria:** 2–4× speedup on tool-calling workloads,
   100% valid JSON, measured against the vLLM baseline with p99 latency.

## Limitations

- Short, single-turn prompts (8 weather queries) — deliberately the harshest case
  for speculation; long-output results pending.
- Single GPU, single model family (Qwen2.5-1.5B), greedy decoding, K=5.
- All performance results to date are **negative**; validity results are positive.

## License

[MIT](LICENSE)
