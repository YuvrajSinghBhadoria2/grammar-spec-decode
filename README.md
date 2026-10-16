# Grammar-Aware Speculative Decoding

**Making speculative decoding work under JSON-schema constraints — with honest, reproducible benchmarks.**

> **Status:** active research. Four instrumented iterations complete (v0–v3).
> v0–v2 are honest negatives that diagnosed the problem; **v3 is positive** —
> grammar-masked EAGLE3 speculative decoding beats both unmasked EAGLE3 and
> no-speculation decoding on an agentic JSON workload (1.19× / 1.15× TPOT,
> +36.5% draft acceptance). See [Results](#results-at-a-glance).

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

Three strategies were implemented, benchmarked end-to-end, and root-caused —
each one ruling out a dead end until the architecture pointed at EAGLE:

| Iteration | Strategy | Verdict |
|-----------|----------|---------|
| v0 | Truncate invalid drafts | ❌ draft length collapses |
| v1 | Replace invalid drafts via bitmask | ❌ valid ≠ model-likely |
| v2 | Standalone draft model + masked logits | ❌ wrong architecture → must share target forward pass (EAGLE) |
| **v3** | **Grammar-masked EAGLE3 draft logits** | ✅ **+36.5% acceptance, 1.19× TPOT vs unmasked EAGLE3** |

## Results at a glance

**v3 (positive result)** — Qwen2.5-7B-Instruct (bf16) target + EAGLE3 head
(`ruipeterpan/...UltraChat`) on **NVIDIA A100**, vLLM 0.11.2, xgrammar
bitmask applied to draft logits at every speculative position, K=5,
batch=1, 12 distinct agentic JSON prompts, 3 timing iterations with
rotated config order (mean ± std). Lower TPOT is better.

| Config | Valid JSON | TPOT (ms) | tok/propose | vs no-spec | vs baseline |
|--------|:----------:|----------:|------------:|-----------:|------------:|
| no_spec | 33/36 | 11.09 ± 0.01 | — | 1.00× | — |
| baseline EAGLE3 (unmasked) | 33/36 | 11.47 ± 0.04 | 1.443 | 0.967× | 1.00× |
| **EAGLE3 + grammar mask** | 33/36 | **9.61 ± 0.03** | **1.969 (+36.5%)** | **1.154×** | **1.193×** |

Raw numbers: [v3-eagle3/](v3-eagle3/) (`RESULTS.md`, `eagle3_grammar_b1_results.json`,
`eagle3_grammar_diag_results.json`).

**v0–v2 (diagnostic negatives)** — Qwen2.5-1.5B-Instruct (bf16) target on
the same A100, vLLM 0.11.2, xgrammar FSM, K=5, 8 JSON-schema-constrained
prompts. Lower TPOT is better.

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
5. **Sharing the target's forward pass is what makes grammar-awareness pay.**
   v3 applies the xgrammar bitmask to EAGLE3 draft logits *after* the shared
   `compute_logits` scatter, at every draft position: 39.1% of draft argmaxes
   change under the mask, acceptance rises from 1.443 → 1.969 tokens/propose
   (+36.5%), and TPOT falls 19.3% vs unmasked EAGLE3 — turning a net-negative
   speculative config (0.967× vs no-spec) into a net-positive one (1.154×).
6. **JSON grammars permit degeneration loops.** One of 12 prompts entered a
   grammatically-legal `" \n\n "` whitespace loop until the token cap —
   structure constraints alone can't stop repetition collapse (equal across
   all configs; see v3 probe).

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
├── v2-draft-model/              ← strategy 3: standalone grammar-masked drafter
└── v3-eagle3/                   ← strategy 4: grammar-masked EAGLE3 (positive)
    ├── eagle3_grammar_b1.py     ← benchmark + class-level EagleProposer patch
    ├── RESULTS.md               ← numbers, methodology, implementation facts
    ├── eagle3_grammar_b1_results.json     ← raw 3-iteration results
    ├── eagle3_grammar_diag_results.json   ← headroom diagnostic
    └── probe_validity.py/.log   ← per-prompt validity probe
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

# v3: grammar-masked EAGLE3 (A100-class GPU; downloads 7B target + EAGLE3 head)
python v3-eagle3/eagle3_grammar_b1.py 3 all     # timing: 3 iterations, rotated order
python v3-eagle3/eagle3_grammar_b1.py 1 grammar # headroom diagnostic (timing invalid)
```

v0–v2 compare `no_spec` / `baseline_ngram` / `grammar_aware` (+ `draft_model_grammar`
in v2). v3 compares `no_spec` / `baseline_eagle3` / `eagle3_grammar_masked`,
gated by flag files (`od_enable`, `od_diag`) so timing runs carry zero
diagnostic GPU syncs.

## Next steps

1. **Robustness:** additional schemas (tool-call grammars with recursion),
   batch > 1, and longer agentic traces to map where the 1.19× holds.
2. **GPU-resident FSM:** xgrammar runs on CPU; one small D2H sync per draft
   position remains — a GPU matcher would shrink the masking overhead further.
3. **Upstream contribution:** vLLM's structured-output + spec-decode paths
   don't compose today (roadmap item: *tool calling + structured outputs*);
   the v3 patch is a candidate upstream design.

## Limitations

- v0–v2 used short, single-turn prompts (8 weather queries) — deliberately the
  harshest case for speculation; v3 moved to 12 distinct agentic prompts
  (~150 output tokens each), but all runs are still batch=1, single GPU
  (A100), greedy decoding, K=5.
- v3 covers one target/head pair (Qwen2.5-7B + its EAGLE3 head) and one
  JSON schema class; generalization across models/schemas/batches is
  untested.
- v3's 3 iterations replicate timing, not output distribution (temp=0).
  Results are self-measured, not yet independently re-verified.
- One prompt produced a grammatical whitespace-loop degeneration under
  every config (33/36 valid each — equal across configs, not a
  spec-decode effect).

## License

[MIT](LICENSE)
