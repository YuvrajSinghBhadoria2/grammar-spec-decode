# Draft blog post — for review before publishing to your site

> Title options: "Speculative Decoding Is Broken Under JSON Schemas — Here's
> the Fix" / "84.9% of Speculative Drafts Are Invalid. Fixing That."
> Suggested tags: speculative decoding, vLLM, structured outputs, EAGLE, xgrammar

---

Every AI agent you ship runs on JSON. Tool calls, structured outputs, schema-
constrained decoding — that's the workload. And speculative decoding, the
standard 2× lever for LLM throughput, quietly falls apart on exactly that
workload. I measured it, root-caused it, and fixed it. Here's the whole
story, including the dead ends.

## The problem, measured

Speculative decoding drafts tokens cheaply (with a small head or model) and
verifies them against the target model in one batched forward pass. When you
add grammar-constrained decoding — the thing that *enforces* schema-valid
output from your agents — the verifier only accepts grammar-valid tokens.
But the draft model knows nothing about the grammar.

Measured on vLLM 0.11.2 with a Qwen2.5-7B target on an A100:

- **84.9% of n-gram drafts were rejected by the grammar FSM** before I even
  got to learned drafts (292/344).
- Unmasked **EAGLE3** — the best draft head vLLM supports — accepted only
  **1.44 draft tokens per step** on agentic JSON, making speculation
  **0.97× slower than no speculation at all**. Spec decode wasn't just
  unhelpful; it was a pessimization.

This is on vLLM's own roadmap as an open item ("tool calling + structured
outputs"), which is how I got interested.

## Three dead ends (kept for honesty)

1. **Truncate invalid drafts.** Valid output, but draft length collapses —
   0.24× speed. Dead.
2. **Replace invalid drafts with the "first valid token" from the bitmask.**
   Grammar-valid, but not model-likely: 83% of drafts got substituted and
   then rejected at verification anyway. Valid ≠ probable. Dead.
3. **Run a separate small draft model with masked logits inside `propose()`.**
   Correct architecture on paper; 61× slower in practice — a separate model
   forward pass inside the sampling loop serializes everything. Dead.

The lesson from all three: the fix has to live *inside the shared forward
pass*, at the point where draft logits already exist.

## The fix: mask the draft's logits, not just the target's

vLLM's EAGLE3 path computes draft logits and immediately argmaxes them.
The head's logits are scattered into the *target's* vocabulary space before
that argmax — which means a standard xgrammar bitmask applies directly, right
there, at every speculative position:

```
draft hidden states → compute_logits (target vocab) → [MASK] → argmax
```

The mask needs a per-request grammar FSM that advances with the drafted
tokens during the K draft steps. So the prototype keeps one matcher per
request, advances it with each draft, and — because of a nasty xgrammar
gotcha below — builds a **fresh matcher per propose call** and replays the
committed-token history instead of relying on `rollback()`.

## Results (A100, batch=1, vLLM 0.11.2, K=5, 12 distinct prompts, 3 timing
iterations, rotated order, diagnostics off in the hot path)

Weather-tool JSON schema (7B target + EAGLE3 head):

| | TPOT | tok/propose | vs no-spec | vs baseline |
|---|---|---|---|---|
| no spec | 11.09 ± 0.01 ms | — | 1.00× | — |
| EAGLE3 unmasked | 11.47 ± 0.04 | 1.443 | 0.97× | — |
| **EAGLE3 + grammar mask** | **9.61 ± 0.03** | **1.969 (+36.5%)** | **1.15×** | **1.19×** |

It wins at every K I tried (3/5/7: 1.15×/1.19×/1.17×), and the best absolute
configuration is K=3 + masking: **1.25× over no speculation**.

Enum-heavy tool-call schema (the harder case; 8 prompts, single session):

| | Valid JSON | TPOT | tok/propose |
|---|---|---|---|
| no spec | 5/8 | 11.02 ms | — |
| EAGLE3 unmasked | **3/8** | 13.54 ms | 1.203 |
| **+ grammar mask** | **5/8** | **10.37 ms (1.31× vs baseline)** | **1.807 (+50.2%)** |

Two things jump out. First, unmasked spec decode doesn't just slow down on
hard schemas — it **hurts validity** (3/8 vs 5/8), pushing decoding into
grammatically-legal degeneration loops. Masking restores validity to the
no-spec level while also being the fastest configuration. Second, the
acceptance boost is *larger* on the schema with more enums (+50% vs +36%),
which matches the mechanism: enums are exactly where unconstrained drafts
wander off.

## The xgrammar landmine

I lost a day to this one, so it's going in writing: **`GrammarMatcher.rollback()`
in xgrammar 0.1.25 is unsound when a permanent accepted prefix precedes the
rolled-back window** and the window contains forced-continuation tokens
(enum values split into `'t','o','o','l'`-style pieces trigger it). The
matcher accepts your rollback call and silently desyncs. Symptom: every
draft gets rejected forever (tok/propose pinned at exactly 1.000) and FSM
replay rejections spike. The robust design is fresh-matcher-per-propose with
history replay — slightly more CPU work per step, zero rollback calls,
rejections 0/2387.

## What's next

The prototype is a monkey-patched `EagleProposer` — the shape of a real vLLM
change is clear (bitmask hook in the draft loop + per-request matcher
lifecycle). I'm writing this up as a design issue upstream. Open questions
I'd like to push on: batch > 1, a GPU-resident FSM to kill the remaining
per-position host sync, and whether the same mask idea applies to MTP heads.

All benchmarks, probes, raw JSON, and the negative results are in the repo
(https://github.com/YuvrajSinghBhadoria2/grammar-spec-decode) — reproduce
with one command per config.

---

*[Numbers are self-measured on one A100, one model pair, two schemas,
greedy decoding, batch=1. Independent re-verification and the full
methodology notes (including why diagnostics were gated out of timing
runs) are in the repo.]*
