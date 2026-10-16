# Grammar-Aware Speculative Decoding

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![vLLM 0.11.2](https://img.shields.io/badge/vLLM-0.11.2-blue)
![xgrammar 0.1.25](https://img.shields.io/badge/xgrammar-0.1.25-blue)
![Benchmarked on A100](https://img.shields.io/badge/Benchmarked-A100%2080GB-green)

**Make speculative decoding pay off under JSON schemas — measured end-to-end on vLLM, with every negative result kept.**

> **Status: research complete.** Five instrumented iterations (v0–v4).
> Three honest negatives diagnosed why spec decode breaks under grammar constraints;
> **v3/v4 fix it** — grammar-masked EAGLE3 drafting wins on two schema families
> (**1.19×** and **1.31×** TPOT vs unmasked EAGLE3), at every draft length tested
> (K ∈ {3, 5, 7}), and *restores* the output validity that constrained spec decode
> loses on hard schemas.

## TL;DR

- **Problem:** under grammar-constrained decoding, most speculative drafts are
  grammatically invalid — **84.9%** rejected for n-gram drafts; EAGLE3 acceptance
  collapses to 1.44 tok/propose, making speculation **slower than no speculation** (0.97×).
- **Fix:** apply the xgrammar bitmask to **draft logits inside the proposer**
  (after `compute_logits`, before `argmax`) — not only at target verification.
- **Result:** draft acceptance **+36.5% → +50.2%** across two schema families;
  TPOT **1.15–1.31×** faster than unmasked EAGLE3; wins at K = 3, 5, 7;
  validity on the hard schema restored from **3/8 → 5/8** (no-spec level).
- **Evidence:** raw JSON, full logs, discarded experiments, and root-cause probes
  are all in this repo — including the first tool-schema run we invalidated and why.

## The problem

Speculative decoding drafts tokens cheaply with a lightweight drafter, then
verifies them against the target model in one batched pass. Under
**grammar-constrained decoding** (JSON schemas, tool calling, structured outputs),
the verifier only accepts grammar-valid tokens — but the drafter knows nothing
about the grammar, so drafts get rejected and speculation stops paying off.

Measured in v0: under a JSON schema, **84.9% of n-gram drafts were rejected by the
grammar FSM** (292 of 344). Unmasked EAGLE3 — the strongest drafter vLLM supports —
accepted only **1.44 tokens per propose call** on agentic JSON and ran
**0.97× slower than no speculation at all**.

This is on [vLLM's Q3 2026 roadmap](https://github.com/vllm-project/vllm/issues/48168)
as an open item (SIG Spec Decode):

> *Improve robustness with tool calling + reasoning and structured outputs.*

## Approach

Draft **with** the grammar instead of against it. Each iteration was benchmarked
end-to-end and root-caused before moving on — dead ends are kept as evidence:

| Iteration | Strategy | Verdict |
|-----------|----------|---------|
| v0 | Truncate grammatically invalid drafts | ❌ valid, but draft length collapses — 0.24× vs no-spec |
| v1 | Substitute invalid drafts with the first bitmask-valid token | ❌ valid ≠ model-likely — 83.1% substituted, then rejected anyway |
| v2 | Standalone 0.5B drafter with masked logits inside `propose()` | ❌ wrong architecture — 61× slower; the drafter must share the target's forward pass |
| **v3** | **Grammar-masked EAGLE3 draft logits** | ✅ **+36.5% acceptance, 1.19× TPOT vs unmasked EAGLE3** |
| **v4** | **Fresh matcher per propose (no `rollback()`) + schema-write fix** | ✅ **validates on a 2nd schema: 1.31×, +50.2%, FSM rejections 0/2387** |

## How it works

vLLM's EAGLE3 path already scatters draft-vocab logits into the **target's**
vocabulary space (`d2t`) before argmax — so a standard xgrammar bitmask applies
directly, at every speculative position:

```
draft hidden states → compute_logits (target vocab) → [grammar mask] → argmax
```

The mask needs grammar state that advances with each drafted token. The v4 design
builds a **fresh `GrammarMatcher` per propose call**, replays committed-token
history into it, masks all K draft positions, then discards it — zero
`rollback()` calls (xgrammar 0.1.25's `rollback()` desynced after a permanent
accepted prefix in our measurements; details in
[v3-eagle3/RESULTS.md](v3-eagle3/RESULTS.md)).

If the EAGLE3 draft-vocab subset contains no grammar-valid token, the fallback
takes the first grammar-valid token — measured fallback rate: **0/4486 masks**.

## Results at a glance

### Headline — v3/v4, agentic JSON (weather-tool schema)

Qwen2.5-7B-Instruct (bf16) + `ruipeterpan/Qwen2.5-7B-Instruct_EAGLE3_UltraChat`
head, NVIDIA A100, vLLM 0.11.2, K=5, batch=1, 12 distinct prompts,
3 timing iterations with rotated config order (mean ± std), diagnostics gated
out of the hot path. Lower TPOT is better.

| Config | Valid JSON | TPOT (ms) | tok/propose | vs no-spec | vs baseline |
|--------|:----------:|----------:|------------:|-----------:|------------:|
| no_spec | 33/36 | 11.09 ± 0.01 | — | 1.00× | — |
| baseline EAGLE3 (unmasked) | 33/36 | 11.47 ± 0.04 | 1.443 | 0.967× | 1.00× |
| **EAGLE3 + grammar mask** | 33/36 | **9.61 ± 0.03** | **1.969 (+36.5%)** | **1.154×** | **1.193×** |

*`tok/propose` = output tokens per proposer call — an acceptance proxy (higher is better).
The 33/36 invalid prompt is a grammatical whitespace-loop degeneration, identical
under every config (probe: [v3-eagle3/probe_validity.log](v3-eagle3/probe_validity.log)).*

### Robustness — K-sweep + second (harder) schema

| Check | Result |
|---|---|
| K-sweep, weather schema | wins at K=3 (**1.149×**), K=5 (1.193×), K=7 (1.166×) vs unmasked; best absolute: **K=3 + mask = 1.25× vs no-spec** |
| Enum-heavy tool-call schema (8 prompts) | **1.306×** vs unmasked EAGLE3, acceptance 1.203 → **1.807 (+50.2%)**, FSM replay rejections **0/2387** |
| Validity on the hard schema | constrained spec decode *degrades* it (baseline **3/8** vs no-spec 5/8); grammar masking **restores 5/8** |
| `GrammarMatcher.rollback()` (xgrammar 0.1.25) | unsound after a permanent accepted prefix when the window holds forced-continuation tokens → v4 uses a **fresh matcher per propose** |

Raw numbers: [v3-eagle3/RESULTS.md](v3-eagle3/RESULTS.md) (methodology + full
protocol), `robustness_results.json`, `tool_fixed_results.json`,
`eagle3_grammar_b1_results.json`.

### Diagnostic negatives (v0–v2)

Qwen2.5-1.5B-Instruct (bf16), same A100, K=5, 8 constrained prompts:

| Iteration | Strategy | Valid | TPOT | vs no-spec | Verdict |
|-----------|----------|:-----:|-----:|:----------:|---------|
| [v0](v0/) | Truncate invalid drafts | 8/8 | 13.5 ms | 0.24× | ❌ draft length collapses |
| [v1](v1-bitmask/) | Bitmask substitution | 8/8 | 13.6 ms | 0.23× | ❌ valid ≠ model-likely |
| [v2](v2-draft-model/) | Standalone draft model | 8/8 | 183.5 ms | 0.02× | ❌ wrong architecture |

## Key findings

1. **The problem is real and large.** 84.9% of n-gram drafts are grammar-invalid
   under a JSON schema — natural-language history doesn't contain schema-valid
   continuations.
2. **Grammar-valid ≠ model-likely.** First-valid-token substitution replaced 83.1%
   of drafts and verification still rejected them (v1).
3. **A drafter must share the target's forward pass.** A standalone 0.5B drafter
   inside `propose()` serialized both models: 61× slower (v2).
4. **Sharing the forward pass is what makes grammar-awareness pay.** Masking EAGLE3
   logits after the `d2t` scatter changes **39.1%** of draft argmaxes, lifts
   acceptance 1.443 → 1.969 tok/propose, and flips a net-negative speculative
   config (0.967× vs no-spec) into a net-positive one (1.154×).
5. **The harder the schema, the larger the acceptance win** — +36.5% on the
   weather schema vs **+50.2%** on the enum-heavy tool schema: enums are exactly
   where unconstrained drafts wander off.
6. **Constrained spec decode can degrade output validity.** On the hard schema,
   unmasked EAGLE3 produced 3/8 valid outputs vs 5/8 for no-spec (degeneration
   loops); grammar-masked drafting restored 5/8.
7. **JSON grammars permit degeneration loops.** Grammatically-legal whitespace/code
   repetition until the token cap is a constrained-decoding problem in general —
   no config can fix it with structure constraints alone.

## Repository layout

```
grammar-spec-decode/
├── README.md, LICENSE
├── v0/                          # strategy 1: truncate invalid drafts
│   ├── grammar_aware_proposer.py
│   ├── benchmark_e2e.py         # 3-config end-to-end benchmark
│   ├── test_grammar_proposer.py # 4 unit tests (CPU-only, no GPU)
│   ├── RESULTS.md               # numbers + root-cause analysis
│   └── benchmark_results.json   # raw run output
├── v1-bitmask/                  # strategy 2: bitmask substitution
├── v2-draft-model/              # strategy 3: standalone masked drafter
└── v3-eagle3/                   # strategies 4+5 (positive results)
    ├── eagle3_grammar_b1.py     # v3 benchmark + class-level EagleProposer patch
    ├── eagle3_grammar_fresh.py  # v4: fresh matcher per propose (no rollback)
    ├── robustness.py            # K-sweep runner (Part B superseded, see RESULTS)
    ├── rerun_tool_fixed.py      # tool-schema rerun after the schema-write fix
    ├── probe_*.py + *.log       # validity probe + clobber root-cause chain
    ├── RESULTS.md               # methodology, corrections, honest negatives
    ├── *_results.json + *.log   # raw evidence for every claim
    ├── VLLM-ISSUE-DRAFT.md      # upstream design issue (drafted)
    └── BLOG-DRAFT.md            # narrative write-up (drafted)
```

## Reproducing

```bash
pip install "vllm==0.11.2" "xgrammar==0.1.25" numpy torch

# CPU-only unit tests (no GPU required)
python v0/test_grammar_proposer.py

# Diagnostic benchmarks v0–v2 (NVIDIA GPU)
python v0/benchmark_e2e.py
python v1-bitmask/benchmark_e2e.py
python v2-draft-model/benchmark_e2e.py

# v3 headline (GPU; downloads the 7B target + EAGLE3 head on first run)
python v3-eagle3/eagle3_grammar_b1.py 3 all      # 3 iterations, rotated order
python v3-eagle3/eagle3_grammar_b1.py 1 grammar  # headroom diagnostic (timing invalid)

# v4: same CLI, fresh-matcher lifecycle
python v3-eagle3/eagle3_grammar_fresh.py 3 all

# Robustness: K-sweep (Part A). Part B predates the schema-write fix and is
# superseded by the fixed rerun below — both are documented in RESULTS.md.
python v3-eagle3/robustness.py
python v3-eagle3/rerun_tool_fixed.py             # tool-schema numbers
```

v0–v2 compare `no_spec` / `baseline_ngram` / `grammar_aware`; v3/v4 compare
`no_spec` / `baseline_eagle3` / `eagle3_grammar_masked`, gated by flag files
(`od_enable`, `od_diag`) so timing runs carry **zero diagnostic GPU syncs**.
Tested on Linux, Python 3.13, A100 80GB.

## Limitations

- All runs: batch=1, single A100, greedy decoding, one target/head pair
  (Qwen2.5-7B + its EAGLE3 head), two JSON schema classes. Generalization across
  models, schemas, and batch sizes is untested.
- Speedups are measured against *this* workload: short agentic JSON outputs
  (~150–370 tokens). Spec decode needs longer outputs to amortize verification;
  the 2–4× figures from the broader literature were **not** reproduced here
  (best: 1.25× vs no-spec).
- xgrammar runs on CPU: one small host sync per draft position remains
  (a GPU-resident matcher would remove it).
- The `rollback()` unsoundness was observed in-session (xgrammar 0.1.25); the
  micro-test transcripts weren't archived — v4 sidesteps it entirely, and a
  minimal repro is available on request.
- Timing numbers were re-derived independently from the raw JSONs post-session
  (all confirmed) — but not yet reproduced by a third party.

## Next steps

1. **Batch > 1** — the bitmask indices path (`apply_token_bitmask_inplace(..., indices=...)`)
   is implemented but untested beyond batch=1.
2. **GPU-resident FSM** — remove the remaining per-position host sync.
3. **MTP heads** — the same mask idea should transfer to vLLM's MTP proposer.
4. **Upstream contribution** — a design issue against the open roadmap item is
   drafted in [v3-eagle3/VLLM-ISSUE-DRAFT.md](v3-eagle3/VLLM-ISSUE-DRAFT.md).

## License

[MIT](LICENSE)
