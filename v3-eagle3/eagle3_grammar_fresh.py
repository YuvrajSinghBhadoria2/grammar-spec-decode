"""EAGLE3 grammar-masked speculative decoding — v4 fresh-matcher variant.

CHANGE vs v3 (eagle3_grammar_b1.py): xgrammar 0.1.25 GrammarMatcher.rollback()
fails to restore state when permanent accepted tokens precede the rolled-back
window (proven standalone; breaks enum-heavy grammars that trigger
jump-forward bookkeeping). v4 therefore creates a FRESH matcher per propose
call, replays the request's committed-token history, advances it with drafted
tokens during the K positions, and DISCARDS it at propose exit — no rollback.
Warmup is now grammar-constrained (v3 warmup was unconstrained).

Same timing methodology as v3: zero diagnostic syncs in the hot path.
Usage: python3 eagle3_grammar_fresh.py [iters] [all|grammar]
Flag files: od_enable (grammar masking on), od_diag (diagnostics on).
"""

from __future__ import annotations

import json
import os
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
import sys
import time

AGENTIC_SCHEMA = {
    "type": "object",
    "properties": {
        "tool": {
            "type": "string",
            "description": "Name of the tool to call",
        },
        "arguments": {
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "units": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                "days": {"type": "integer"},
                "include_alerts": {"type": "boolean"},
            },
            "required": ["location", "units", "days"],
        },
        "reasoning": {"type": "string"},
        "confidence": {"type": "number"},
    },
    "required": ["tool", "arguments", "reasoning", "confidence"],
}

# 12 distinct prompts -> independent structured generations under temp=0.
AGENTIC_PROMPTS = [
    (
        "You are an AI agent. The user asks: 'What's the weather forecast for "
        "Tokyo, Paris, and New York over the next 3 days? I need it in Celsius "
        "and want to know about any weather alerts.' Call the appropriate "
        "weather tool with the right arguments. Think step by step about what "
        "location, units, and number of days to request, then output your tool "
        "call as JSON."
    ),
    (
        "You are an AI agent. A pilot requests: 'Give me the forecast for "
        "London and Berlin for the next 5 days in Fahrenheit, including "
        "alerts.' Decide the right weather tool call, reason step by step "
        "about the arguments, and output only the JSON tool call."
    ),
    (
        "You are an AI agent. A farmer asks: 'Will it rain in Delhi over the "
        "coming 7 days? Show temperatures in Celsius and flag any alerts.' "
        "Think step by step, then emit the weather tool call as JSON."
    ),
    (
        "You are an AI agent. A user plans a wedding outdoors in Sydney: "
        "'What will the weather be like there for the next 4 days in "
        "Celsius, and are there alerts?' Reason carefully, then output the "
        "tool call as JSON."
    ),
    (
        "You are an AI agent. A logistics manager asks: 'Weather for Chicago "
        "and Houston for 2 days, Fahrenheit, with alerts included.' Think "
        "step by step about arguments and respond with the JSON tool call."
    ),
    (
        "You are an AI agent. A hiker asks: 'Forecast for Reykjavik over the "
        "next 6 days in Celsius, plus any weather alerts.' Reason about the "
        "arguments first, then output the JSON tool call only."
    ),
    (
        "You are an AI agent. A sailor requests: 'Conditions in Miami for the "
        "next 3 days, Fahrenheit, alerts on.' Think step by step, then "
        "produce the weather tool call as JSON."
    ),
    (
        "You are an AI agent. A developer in Singapore asks: 'Weather there "
        "for 5 days in Celsius with alert status.' Consider the arguments "
        "carefully, then output the JSON tool call."
    ),
    (
        "You are an AI agent. A parent asks: 'Is it cold in Moscow for the "
        "next 2 days if shown in Fahrenheit? Include alerts.' Reason step by "
        "step and emit the JSON tool call."
    ),
    (
        "You are an AI agent. An event organizer asks: 'Forecast for Cairo "
        "over 4 days in Celsius, with any alerts.' Think through the "
        "arguments, then output only the JSON weather tool call."
    ),
    (
        "You are an AI agent. A student asks: 'Weather in São Paulo for the "
        "next 3 days, Celsius, and whether alerts apply.' Reason first, then "
        "write the tool call as JSON."
    ),
    (
        "You are an AI agent. A truck dispatcher asks: 'Give me the forecast "
        "for Toronto and Vancouver for 3 days in Fahrenheit including "
        "alerts.' Think step by step about the arguments and respond with "
        "the JSON tool call."
    ),
]

SCHEMA_PATH = os.path.join(_BASE_DIR, "agentic_schema.json")
STATS_PATH = os.path.join(_BASE_DIR, "grammar_stats.json")
BASE_STATS_PATH = os.path.join(_BASE_DIR, "base_stats.json")
ENABLE_PATH = os.path.join(_BASE_DIR, "od_enable")
DIAG_PATH = os.path.join(_BASE_DIR, "od_diag")
TOKENIZER_NAME = "Qwen/Qwen2.5-7B-Instruct"

# NOTE: do NOT write SCHEMA_PATH at module import — the engine worker re-imports
# this module (spawn) and would clobber a schema written by the parent for a
# different config. Callers must write SCHEMA_PATH before creating the engine.


def _grammar_enabled() -> bool:
    return os.path.exists(ENABLE_PATH)


def _diag_enabled() -> bool:
    return os.path.exists(DIAG_PATH)


def _install_grammar_patch():
    """Unconditional class-level monkey-patch of EagleProposer (reaches the
    engine worker). Masking gated at propose-time by ENABLE flag file;
    diagnostics gated by DIAG flag file (never present during timing runs)."""
    import torch
    import xgrammar as xgr
    from vllm.v1.spec_decode.eagle import EagleProposer

    if getattr(EagleProposer, "_od_grammar_patched", False):
        return
    EagleProposer._od_grammar_patched = True

    _init = EagleProposer.__init__
    _propose = EagleProposer.propose
    _prep_padded = EagleProposer.prepare_next_token_ids_padded
    _prep_cpu = EagleProposer.prepare_next_token_ids_cpu

    def patched_init(self, *a, **kw):
        _init(self, *a, **kw)
        self._g_matchers = {}
        self._g_hist = {}
        self._g_logged_fail = set()
        self._g_cg = None
        self._g_ti = None
        self._g_words = None
        self._g_build_s = None
        self._g_bitmask = None
        self._g_committed = None
        self._g_pos = 0
        self._g_advanced = None
        self._g_req_ids = None
        self._g_batch = 0
        self._g_diag = False
        self._g_stats = {
            "propose_calls": 0,
            "mask_applied": 0,
            "mask_rows": 0,
            "mask_changed": 0,
            "fallback_first_valid": 0,
            "draft_advances": 0,
            "accepted_committed": 0,
            "rejected_committed": 0,
            "terminated": 0,
        }

    def _ensure_built(self):
        if self._g_cg is not None:
            return
        from transformers import AutoTokenizer

        t0 = time.perf_counter()
        with open(SCHEMA_PATH) as f:
            schema = json.load(f)
        tok = AutoTokenizer.from_pretrained(TOKENIZER_NAME, trust_remote_code=True)
        ti = xgr.TokenizerInfo.from_huggingface(tok)
        cg = xgr.GrammarCompiler(ti).compile_grammar(
            xgr.Grammar.from_json_schema(json.dumps(schema))
        )
        self._g_ti = ti
        self._g_cg = cg
        self._g_tok = tok
        self._g_words = (ti.vocab_size + 31) // 32
        self._g_build_s = time.perf_counter() - t0

    def _bitmask_for(self, batch):
        if self._g_bitmask is None or self._g_bitmask.shape[0] < batch:
            self._g_bitmask = torch.zeros((batch, self._g_words), dtype=torch.int32)
        else:
            self._g_bitmask[:batch].zero_()
        return self._g_bitmask

    def _fallback(self, logits, all_neginf, bm, active):
        # Optimization only: if the draft subset has no grammar-valid token,
        # pick the first grammar-valid token instead of argmax over all -inf.
        # Worst case without it: a wasted draft slot (target rejects it);
        # output validity is unaffected either way.
        words = bm.numpy()
        for i in active:
            if not bool(all_neginf[i]):
                continue
            row = words[i]
            found = -1
            for w_idx in range(len(row)):
                word = int(row[w_idx]) & 0xFFFFFFFF
                if word == 0:
                    continue
                for b in range(32):
                    if word & (1 << b):
                        tok_id = w_idx * 32 + b
                        if tok_id < self._g_ti.vocab_size:
                            found = tok_id
                            break
                if found >= 0:
                    break
            if found >= 0:
                logits[i, found] = 0.0
                self._g_stats["fallback_first_valid"] += 1

    def _fill_mask(self, logits, batch):
        req_ids = self._g_req_ids
        bm = self._g_bitmask
        active = []
        for i in range(batch):
            rid = req_ids[i] if i < len(req_ids) else None
            m = self._g_matchers.get(rid)
            if m is None:
                continue
            if m.is_terminated():
                self._g_stats["terminated"] += 1
                continue
            if m.fill_next_token_bitmask(bm, i):
                active.append(i)
        if not active:
            return
        self._g_stats["mask_applied"] += 1
        self._g_stats["mask_rows"] += len(active)
        bm_gpu = bm[:batch].to(logits.device, non_blocking=True)
        xgr.apply_token_bitmask_inplace(logits, bm_gpu, indices=active)
        if self._g_diag:
            all_neginf = (logits == float("-inf")).all(dim=-1)
            if bool(all_neginf.any()):
                _fallback(self, logits, all_neginf, bm, active)

    def patched_prep_padded(self, cam, sampled_token_ids, requests, gib, dri, ndr):
        out = _prep_padded(self, cam, sampled_token_ids, requests, gib, dri, ndr)
        if not _grammar_enabled():
            self._g_committed = None
            return out
        _, valid_count = out
        try:
            sc = sampled_token_ids.detach().cpu().tolist()
            vc = valid_count.detach().cpu().tolist()
            committed = []
            for i in range(len(sc)):
                n = int(vc[i])
                committed.append([int(t) for t in sc[i][:n] if t >= 0])
            self._g_committed = committed
        except Exception:
            self._g_committed = None
        return out

    def patched_prep_cpu(self, sampled_token_ids, requests, gib, nst):
        out = _prep_cpu(self, sampled_token_ids, requests, gib, nst)
        if not _grammar_enabled():
            self._g_committed = None
            return out
        try:
            committed = []
            for arr in sampled_token_ids:
                committed.append([int(t) for t in arr if t >= 0])
            self._g_committed = committed
        except Exception:
            self._g_committed = None
        return out

    def patched_propose(
        self,
        target_token_ids,
        target_positions,
        target_hidden_states,
        next_token_ids,
        last_token_indices,
        common_attn_metadata,
        sampling_metadata,
        mm_embed_inputs=None,
    ):
        if not _grammar_enabled():
            # Baseline path: count propose calls only (pure counter, no GPU).
            self._g_stats["propose_calls"] += 1
            try:
                with open(BASE_STATS_PATH, "w") as f:
                    json.dump(
                        {"propose_calls": self._g_stats["propose_calls"]}, f
                    )
            except Exception:
                pass
            return _propose(
                self,
                target_token_ids,
                target_positions,
                target_hidden_states,
                next_token_ids,
                last_token_indices,
                common_attn_metadata,
                sampling_metadata,
                mm_embed_inputs,
            )

        self._g_stats["propose_calls"] += 1
        self._g_diag = _diag_enabled()
        _ensure_built(self)
        req_ids = list(self.runner.input_batch.req_ids)
        batch = next_token_ids.shape[0]
        self._g_req_ids = req_ids
        self._g_batch = batch

        committed = self._g_committed or [[] for _ in range(batch)]

        for i in range(batch):
            rid = req_ids[i] if i < len(req_ids) else None
            if rid is None:
                continue
            # v4: FRESH matcher every propose; replay the request's full
            # committed-token history (avoids xgrammar rollback-after-prefix
            # desync). Draft advances during this propose are simply discarded
            # with the matcher at propose exit — no rollback calls at all.
            try:
                m = xgr.GrammarMatcher(
                    self._g_cg, max_rollback_tokens=self.num_speculative_tokens + 1
                )
                self._g_matchers[rid] = m
            except Exception:
                continue
            hist = self._g_hist.setdefault(rid, [])
            toks = committed[i] if i < len(committed) else []
            if not toks and not hist:
                toks = [int(next_token_ids[i].item())]
            hist.extend(toks)
            n_new = len(toks)
            for j, t in enumerate(hist):
                if m.is_terminated():
                    break
                ok = m.accept_token(t)
                if not ok and rid not in self._g_logged_fail:
                    self._g_logged_fail.add(rid)
                    try:
                        with open(os.path.join(_BASE_DIR, "fresh_fail.log"), "a") as f:
                            f.write(
                                f"rid={rid} fail_at_hist_idx={j} hist_len={len(hist)} "
                                f"tok={t} text={self._g_tok.decode([t])!r} "
                                f"hist_head={hist[:12]} n_new={n_new}\n"
                            )
                    except Exception:
                        pass
                if j >= len(hist) - n_new:
                    if ok:
                        self._g_stats["accepted_committed"] += 1
                    else:
                        self._g_stats["rejected_committed"] += 1

        self._g_pos = 0
        self._g_advanced = [0] * batch
        orig_cl = self.model.compute_logits
        gstate = self

        def masked_cl(hidden):
            logits = orig_cl(hidden)
            pos = gstate._g_pos
            gstate._g_pos += 1
            if pos > 0:
                prev = gstate.input_ids[:batch]
                for i in range(batch):
                    rid = req_ids[i] if i < len(req_ids) else None
                    m = gstate._g_matchers.get(rid)
                    if m is None or m.is_terminated():
                        continue
                    t = int(prev[i].item())
                    if m.accept_token(t):
                        gstate._g_advanced[i] += 1
                        gstate._g_stats["draft_advances"] += 1
            _bitmask_for(gstate, batch)
            if gstate._g_diag:
                unmasked = logits.argmax(dim=-1)
                _fill_mask(gstate, logits, batch)
                masked = logits.argmax(dim=-1)
                gstate._g_stats["mask_changed"] += int(
                    (unmasked != masked).sum().item()
                )
            else:
                _fill_mask(gstate, logits, batch)
            return logits

        self.model.compute_logits = masked_cl
        try:
            draft = _propose(
                self,
                target_token_ids,
                target_positions,
                target_hidden_states,
                next_token_ids,
                last_token_indices,
                common_attn_metadata,
                sampling_metadata,
                mm_embed_inputs,
            )
        finally:
            try:
                del self.model.compute_logits
            except AttributeError:
                pass
            # v4: per-propose matchers are discarded as a unit; drafted-token
            # advances died with them — no rollback (xgrammar rollback after
            # a permanent prefix is unsound; see RESULTS).
            self._g_matchers = {}
            self._g_committed = None
            try:
                with open(STATS_PATH, "w") as f:
                    json.dump({**self._g_stats, "build_s": self._g_build_s}, f)
            except Exception:
                pass
        return draft

    EagleProposer.__init__ = patched_init
    EagleProposer.propose = patched_propose
    EagleProposer.prepare_next_token_ids_padded = patched_prep_padded
    EagleProposer.prepare_next_token_ids_cpu = patched_prep_cpu


_install_grammar_patch()


def _set_flag(path: str, on: bool) -> None:
    if on:
        with open(path, "w") as f:
            f.write("1")
    elif os.path.exists(path):
        os.remove(path)


def run_config(
    name: str,
    mode: str,
    target: str,
    spec_config: dict | None,
    diag: bool = False,
) -> dict:
    """mode in {'none', 'eagle3', 'eagle3_grammar'}; 12 distinct prompts.

    diag=False for timing runs (zero diagnostic GPU syncs); diag=True only
    for the dedicated instrumentation run.
    """
    use_spec = mode != "none"

    _set_flag(ENABLE_PATH, mode == "eagle3_grammar")
    _set_flag(DIAG_PATH, diag)

    # Worker re-imports this module at spawn; schema must be written HERE
    # (parent, immediately before engine start), not at module import.
    with open(SCHEMA_PATH, "w") as _sf:
        json.dump(AGENTIC_SCHEMA, _sf)

    # Remove stale stats so a failed patch cannot poison this config's reads.
    for p in (STATS_PATH, BASE_STATS_PATH):
        if os.path.exists(p):
            os.remove(p)

    print("=" * 70, flush=True)
    print(f"CONFIG: {name}  (mode={mode}, diag={diag})", flush=True)
    print("=" * 70, flush=True)

    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    kwargs = dict(
        model=target,
        max_model_len=4096,
        dtype="bfloat16",
        max_num_seqs=1,
        gpu_memory_utilization=0.9,
        enforce_eager=bool(use_spec),
        disable_log_stats=False,
    )
    if use_spec and spec_config:
        kwargs["speculative_config"] = spec_config

    llm = LLM(**kwargs)

    sp = SamplingParams(
        temperature=0.0,
        max_tokens=512,
        structured_outputs=StructuredOutputsParams(json=AGENTIC_SCHEMA),
    )

    if use_spec:
        # Constrained warmup (v3 warmup was unconstrained — polluted the
        # warmup request's FSM with reasoning tokens; harmless for measured
        # output but wrong for clean FSM stats).
        llm.generate(
            [AGENTIC_PROMPTS[0]],
            SamplingParams(
                temperature=0.0,
                max_tokens=32,
                structured_outputs=StructuredOutputsParams(json=AGENTIC_SCHEMA),
            ),
        )

    prompts = AGENTIC_PROMPTS
    sys.stdout.flush()
    start = time.perf_counter()
    outputs = llm.generate(prompts, sp)
    elapsed = time.perf_counter() - start
    sys.stdout.flush()

    valid_count = 0
    total_tokens = 0
    for out in outputs:
        text = out.outputs[0].text
        total_tokens += len(out.outputs[0].token_ids)
        try:
            parsed = json.loads(text)
            if all(k in parsed for k in AGENTIC_SCHEMA["required"]):
                valid_count += 1
        except json.JSONDecodeError:
            pass

    n = len(outputs)
    result = {
        "name": name,
        "mode": mode,
        "valid": valid_count,
        "total": n,
        "total_tokens": total_tokens,
        "avg_tokens": total_tokens / n,
        "elapsed_s": elapsed,
        "tpot_ms": (elapsed / total_tokens * 1000) if total_tokens else 0,
        "tok_per_s": total_tokens / elapsed if elapsed else 0,
    }

    if mode == "eagle3_grammar" and os.path.exists(STATS_PATH):
        try:
            with open(STATS_PATH) as f:
                result["grammar_stats"] = json.load(f)
        except Exception:
            pass
    if mode == "eagle3" and os.path.exists(BASE_STATS_PATH):
        try:
            with open(BASE_STATS_PATH) as f:
                result["base_stats"] = json.load(f)
        except Exception:
            pass

    print(
        f"  [{name}] Valid={valid_count}/{n}  TPOT={result['tpot_ms']:.2f}ms  "
        f"Tok/s={result['tok_per_s']:.1f}  AvgTok={result['avg_tokens']:.1f}  "
        f"proposes={result.get('base_stats', result.get('grammar_stats', {})).get('propose_calls', '-')}",
        flush=True,
    )

    del llm
    import torch

    torch.cuda.empty_cache()
    return result


def _tokens_per_propose(r: dict) -> float | None:
    pc = (r.get("base_stats") or r.get("grammar_stats") or {}).get("propose_calls")
    if pc and pc > 0:
        return r["total_tokens"] / pc
    return None


def _mean_std(xs: list[float]) -> tuple[float, float]:
    m = sum(xs) / len(xs)
    var = sum((x - m) ** 2 for x in xs) / len(xs) if len(xs) > 1 else 0.0
    return m, var**0.5


def run_benchmark(target: str, eagle3_head: str) -> list[dict]:
    configs = [
        ("no_spec_7b_b1", "none", None),
        (
            "baseline_eagle3_b1",
            "eagle3",
            {"method": "eagle3", "model": eagle3_head, "num_speculative_tokens": 5},
        ),
        (
            "eagle3_grammar_masked_b1",
            "eagle3_grammar",
            {"method": "eagle3", "model": eagle3_head, "num_speculative_tokens": 5},
        ),
    ]
    results = []
    for name, mode, spec in configs:
        results.append(run_config(name, mode, target, spec))
    return results


def main() -> None:
    target = "Qwen/Qwen2.5-7B-Instruct"
    eagle3_head = "ruipeterpan/Qwen2.5-7B-Instruct_EAGLE3_UltraChat"

    n_iters = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    scope = sys.argv[2] if len(sys.argv) > 2 else "all"

    # Ensure clean flag state at start (crashed prior runs must not leak).
    _set_flag(ENABLE_PATH, False)
    _set_flag(DIAG_PATH, False)

    print()
    print("EAGLE3 Grammar-Masked SpecDec Benchmark (batch=1)", flush=True)
    print(f"Target: {target} / head: {eagle3_head}", flush=True)
    print(f"12 distinct prompts, warmup=1, iterations={n_iters}, scope={scope}", flush=True)
    print(flush=True)

    if scope == "grammar":
        # Diagnostic run: grammar config only, diagnostics enabled.
        r = run_config(
            "eagle3_grammar_masked_diag",
            "eagle3_grammar",
            target,
            {
                "method": "eagle3",
                "model": eagle3_head,
                "num_speculative_tokens": 5,
            },
            diag=True,
        )
        _set_flag(ENABLE_PATH, False)
        _set_flag(DIAG_PATH, False)
        gs = r.get("grammar_stats", {})
        print(flush=True)
        print("=== DIAGNOSTIC RESULTS (timing NOT valid, diagnostics on) ===", flush=True)
        print(f"grammar_stats: {json.dumps(gs)}", flush=True)
        if gs.get("mask_rows"):
            print(
                f"mask_changed (headroom): {gs.get('mask_changed', 0)}/"
                f"{gs['mask_rows']} = {gs['mask_changed'] / gs['mask_rows']:.3f}",
                flush=True,
            )
        with open(os.path.join(_BASE_DIR, "eagle3_grammar_diag_results.json"), "w") as f:
            json.dump(r, f, indent=2, default=str)
        return

    all_iters: list[list[dict]] = []
    for it in range(n_iters):
        # Rotate config order across iterations to cancel order effects.
        configs = [
            ("no_spec_7b_b1", "none", None),
            (
                "baseline_eagle3_b1",
                "eagle3",
                {"method": "eagle3", "model": eagle3_head, "num_speculative_tokens": 5},
            ),
            (
                "eagle3_grammar_masked_b1",
                "eagle3_grammar",
                {"method": "eagle3", "model": eagle3_head, "num_speculative_tokens": 5},
            ),
        ]
        k = it % len(configs)
        configs = configs[k:] + configs[:k]

        print(flush=True)
        print("#" * 70, flush=True)
        print(f"# ITERATION {it + 1}/{n_iters}  (order: {[c[0] for c in configs]})", flush=True)
        print("#" * 70, flush=True)
        iter_results = []
        for name, mode, spec in configs:
            iter_results.append(run_config(name, mode, target, spec))
        # Restore canonical order for aggregation.
        by_name = {r["name"]: r for r in iter_results}
        iter_results = [
            by_name["no_spec_7b_b1"],
            by_name["baseline_eagle3_b1"],
            by_name["eagle3_grammar_masked_b1"],
        ]
        all_iters.append(iter_results)

    _set_flag(ENABLE_PATH, False)
    _set_flag(DIAG_PATH, False)

    # Aggregate mean +/- std across iterations.
    names = ["no_spec_7b_b1", "baseline_eagle3_b1", "eagle3_grammar_masked_b1"]
    agg = {}
    for nm in names:
        tpots = []
        tpps = []
        for it_res in all_iters:
            r = next(x for x in it_res if x["name"] == nm)
            tpots.append(r["tpot_ms"])
            tp = _tokens_per_propose(r)
            if tp is not None:
                tpps.append(tp)
        m, s = _mean_std(tpots)
        agg[nm] = {"tpot_mean": m, "tpot_std": s, "tpots": tpots}
        if tpps:
            tm, ts = _mean_std(tpps)
            agg[nm]["tpp_mean"] = tm
            agg[nm]["tpp_std"] = ts

    print(flush=True)
    print("=" * 72, flush=True)
    print(f"RESULTS (batch=1, {n_iters} iterations, mean +/- std)", flush=True)
    print("=" * 72, flush=True)
    for nm in names:
        a = agg[nm]
        valid_tot = sum(
            x["valid"] for it_res in all_iters for x in it_res if x["name"] == nm
        )
        n_tot = sum(
            x["total"] for it_res in all_iters for x in it_res if x["name"] == nm
        )
        line = (
            f"{nm:<28} Valid={valid_tot}/{n_tot}  "
            f"TPOT={a['tpot_mean']:.2f}+-{a['tpot_std']:.2f}ms  "
            f"per-iter={[round(x, 2) for x in a['tpots']]}"
        )
        if "tpp_mean" in a:
            line += f"  tok/propose={a['tpp_mean']:.3f}+-{a['tpp_std']:.3f}"
        print(line, flush=True)

    ns, base, gram = (agg[n] for n in names)
    print(flush=True)
    print(
        f"speedup vs no_spec:  baseline={ns['tpot_mean'] / base['tpot_mean']:.3f}x  "
        f"grammar_masked={ns['tpot_mean'] / gram['tpot_mean']:.3f}x",
        flush=True,
    )
    print(
        f"grammar_masked vs baseline_eagle3: "
        f"{base['tpot_mean'] / gram['tpot_mean']:.3f}x",
        flush=True,
    )
    if "tpp_mean" in base and "tpp_mean" in gram:
        print(
            f"acceptance (tokens/propose) ratio gram/base: "
            f"{gram['tpp_mean'] / base['tpp_mean']:.3f}",
            flush=True,
        )

    out = {
        "iterations": n_iters,
        "aggregate": agg,
        "raw": all_iters,
    }
    out_path = os.path.join(_BASE_DIR, "eagle3_grammar_fresh_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(flush=True)
    print(f"Results saved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
