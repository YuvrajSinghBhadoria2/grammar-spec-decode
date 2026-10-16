"""Robustness runs for grammar-masked EAGLE3: second grammar + K-sweep.

Part A: weather schema, K in {3,7} — baseline vs grammar-masked
        (K=5 already measured in v3; K sweep checks the advantage is
        not specific to one K).
Part B: NEW strict tool-call schema (nested enums + arrays), K=5 —
        no_spec / baseline / grammar, same protocol as v3.

Methodology: timing-only (diagnostics off), warmup absorbs grammar
build, 12 weather prompts / 8 tool-call prompts, temp=0.

Usage: python3 robustness.py   (writes robustness_results.json)
"""

from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, "/root/grammar-spec")
from eagle3_grammar_b1 import (  # noqa: E402  (installs EagleProposer patch)
    AGENTIC_PROMPTS,
    AGENTIC_SCHEMA,
    BASE_STATS_PATH,
    DIAG_PATH,
    ENABLE_PATH,
    SCHEMA_PATH,
    STATS_PATH,
    TOKENIZER_NAME,
    _set_flag,
    _tokens_per_propose,
)

TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "tool": {
            "type": "string",
            "enum": ["run_python", "run_javascript", "query_sql", "send_email"],
        },
        "args": {
            "type": "object",
            "properties": {
                "code": {"type": "string"},
                "target": {"type": "string", "enum": ["prod", "staging", "dev"]},
                "timeout_sec": {"type": "integer", "minimum": 1, "maximum": 300},
                "deps": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["pandas", "numpy", "requests", "json"],
                    },
                },
                "dry_run": {"type": "boolean"},
            },
            "required": ["target", "timeout_sec"],
        },
        "explanation": {"type": "string"},
        "risk": {"type": "string", "enum": ["low", "medium", "high"]},
    },
    "required": ["tool", "args", "explanation", "risk"],
}

TOOL_PROMPTS = [
    (
        "You are an AI agent. Run this Python snippet in staging with a 60s "
        "timeout, deps pandas and json, dry run: 'print(df.head())'. Reason "
        "step by step, then output the tool call as JSON."
    ),
    (
        "You are an AI agent. Query production analytics with SQL "
        "'SELECT count(*) FROM events' — 30s timeout, this is risky. Think "
        "first, then output the JSON tool call."
    ),
    (
        "You are an AI agent. Execute JavaScript in the dev sandbox, 10s "
        "timeout: 'console.log(1+1)'. Reason about arguments, then emit the "
        "JSON tool call only."
    ),
    (
        "You are an AI agent. Run a numpy reduction on prod, 120s timeout, "
        "deps [numpy], not a dry run. Think step by step, then output the "
        "JSON tool call."
    ),
    (
        "You are an AI agent. Run SQL on staging: "
        "'SELECT * FROM users LIMIT 10', 15s timeout, deps [json]. Reason "
        "about the arguments, then output the JSON tool call."
    ),
    (
        "You are an AI agent. Send the release notes email to the team — "
        "production target, 20s timeout, low risk. Think first, then output "
        "the JSON tool call."
    ),
    (
        "You are an AI agent. Run this JS in staging, 20s timeout, deps "
        "[requests, json]: 'fetch(url)'. Reason step by step, then output "
        "the JSON tool call."
    ),
    (
        "You are an AI agent. Run a pandas pipeline on prod, 300s timeout, "
        "deps [pandas, numpy], high risk. Think carefully, then output the "
        "JSON tool call."
    ),
]

VALID_TOOLS = {"run_python", "run_javascript", "query_sql", "send_email"}
VALID_RISKS = {"low", "medium", "high"}
VALID_TARGETS = {"prod", "staging", "dev"}
VALID_DEPS = {"pandas", "numpy", "requests", "json"}


def valid_weather(text: str) -> bool:
    try:
        parsed = json.loads(text)
        return all(k in parsed for k in AGENTIC_SCHEMA["required"])
    except json.JSONDecodeError:
        return False


def valid_tool(text: str) -> bool:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return False
    if not all(k in parsed for k in TOOL_SCHEMA["required"]):
        return False
    if parsed.get("tool") not in VALID_TOOLS:
        return False
    if parsed.get("risk") not in VALID_RISKS:
        return False
    args = parsed.get("args")
    if not isinstance(args, dict) or args.get("target") not in VALID_TARGETS:
        return False
    deps = args.get("deps")
    if deps is not None and (
        not isinstance(deps, list) or any(d not in VALID_DEPS for d in deps)
    ):
        return False
    return True


def run_one(
    name: str,
    mode: str,
    spec: dict | None,
    prompts: list[str],
    schema: dict,
    valid_fn,
) -> dict:
    use_spec = mode != "none"
    with open(SCHEMA_PATH, "w") as f:
        json.dump(schema, f)
    _set_flag(ENABLE_PATH, mode == "eagle3_grammar")
    _set_flag(DIAG_PATH, False)
    for p in (STATS_PATH, BASE_STATS_PATH):
        if os.path.exists(p):
            os.remove(p)

    print("=" * 70, flush=True)
    print(f"CONFIG: {name}  (mode={mode})", flush=True)
    print("=" * 70, flush=True)

    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    kwargs = dict(
        model="Qwen/Qwen2.5-7B-Instruct",
        max_model_len=4096,
        dtype="bfloat16",
        max_num_seqs=1,
        gpu_memory_utilization=0.9,
        enforce_eager=bool(use_spec),
        disable_log_stats=False,
    )
    if spec:
        kwargs["speculative_config"] = spec
    llm = LLM(**kwargs)

    sp = SamplingParams(
        temperature=0.0,
        max_tokens=512,
        structured_outputs=StructuredOutputsParams(json=schema),
    )
    if use_spec:
        llm.generate(
            [prompts[0]], SamplingParams(temperature=0.0, max_tokens=32)
        )

    sys.stdout.flush()
    start = time.perf_counter()
    outputs = llm.generate(prompts, sp)
    elapsed = time.perf_counter() - start
    sys.stdout.flush()

    valid = 0
    total_tokens = 0
    for out in outputs:
        total_tokens += len(out.outputs[0].token_ids)
        if valid_fn(out.outputs[0].text):
            valid += 1

    n = len(outputs)
    result = {
        "name": name,
        "mode": mode,
        "valid": valid,
        "total": n,
        "total_tokens": total_tokens,
        "elapsed_s": elapsed,
        "tpot_ms": (elapsed / total_tokens * 1000) if total_tokens else 0,
        "tok_per_s": total_tokens / elapsed if elapsed else 0,
    }
    stats_path = STATS_PATH if mode == "eagle3_grammar" else BASE_STATS_PATH
    if os.path.exists(stats_path):
        with open(stats_path) as f:
            result["stats"] = json.load(f)
    print(
        f"  [{name}] Valid={valid}/{n}  TPOT={result['tpot_ms']:.2f}ms  "
        f"AvgTok={total_tokens / n:.1f}  "
        f"proposes={result.get('stats', {}).get('propose_calls', '-')}",
        flush=True,
    )

    del llm
    import torch

    torch.cuda.empty_cache()
    return result


def spec_cfg(k: int) -> dict:
    return {
        "method": "eagle3",
        "model": "ruipeterpan/Qwen2.5-7B-Instruct_EAGLE3_UltraChat",
        "num_speculative_tokens": k,
    }


def main() -> None:
    results: dict = {"weather_k_sweep": [], "tool_schema_k5": []}

    print("\n### PART A: weather schema, K-sweep", flush=True)
    # K=7: grammar first, then baseline (reverse of v3 order).
    results["weather_k_sweep"].append(
        run_one(
            "grammar_k7", "eagle3_grammar", spec_cfg(7),
            AGENTIC_PROMPTS, AGENTIC_SCHEMA, valid_weather,
        )
    )
    results["weather_k_sweep"].append(
        run_one(
            "baseline_k7", "eagle3", spec_cfg(7),
            AGENTIC_PROMPTS, AGENTIC_SCHEMA, valid_weather,
        )
    )
    # K=3: baseline first, then grammar.
    results["weather_k_sweep"].append(
        run_one(
            "baseline_k3", "eagle3", spec_cfg(3),
            AGENTIC_PROMPTS, AGENTIC_SCHEMA, valid_weather,
        )
    )
    results["weather_k_sweep"].append(
        run_one(
            "grammar_k3", "eagle3_grammar", spec_cfg(3),
            AGENTIC_PROMPTS, AGENTIC_SCHEMA, valid_weather,
        )
    )

    print("\n### PART B: strict tool-call schema, K=5", flush=True)
    results["tool_schema_k5"].append(
        run_one(
            "tool_no_spec", "none", None,
            TOOL_PROMPTS, TOOL_SCHEMA, valid_tool,
        )
    )
    results["tool_schema_k5"].append(
        run_one(
            "tool_baseline_eagle3", "eagle3", spec_cfg(5),
            TOOL_PROMPTS, TOOL_SCHEMA, valid_tool,
        )
    )
    results["tool_schema_k5"].append(
        run_one(
            "tool_grammar_masked", "eagle3_grammar", spec_cfg(5),
            TOOL_PROMPTS, TOOL_SCHEMA, valid_tool,
        )
    )

    _set_flag(ENABLE_PATH, False)
    _set_flag(DIAG_PATH, False)

    # Summary
    print(flush=True)
    print("=" * 72, flush=True)
    print("ROBUSTNESS SUMMARY", flush=True)
    print("=" * 72, flush=True)
    for group, rows in results.items():
        print(f"[{group}]", flush=True)
        for r in rows:
            tpp = _tokens_per_propose(
                {
                    "total_tokens": r["total_tokens"],
                    "base_stats": r.get("stats") if r["mode"] == "eagle3" else None,
                    "grammar_stats": r.get("stats")
                    if r["mode"] == "eagle3_grammar"
                    else None,
                }
            )
            tpp_s = f"  tok/propose={tpp:.3f}" if tpp else ""
            print(
                f"  {r['name']:<24} Valid={r['valid']}/{r['total']}  "
                f"TPOT={r['tpot_ms']:.2f}ms{tpp_s}",
                flush=True,
            )

    # Ratios
    A = {r["name"]: r for r in results["weather_k_sweep"]}
    if "grammar_k7" in A and "baseline_k7" in A:
        print(
            f"K=7 grammar vs baseline: "
            f"{A['baseline_k7']['tpot_ms'] / A['grammar_k7']['tpot_ms']:.3f}x",
            flush=True,
        )
    if "grammar_k3" in A and "baseline_k3" in A:
        print(
            f"K=3 grammar vs baseline: "
            f"{A['baseline_k3']['tpot_ms'] / A['grammar_k3']['tpot_ms']:.3f}x",
            flush=True,
        )
    B = {r["name"]: r for r in results["tool_schema_k5"]}
    if len(B) == 3:
        print(
            f"tool-schema: grammar vs baseline "
            f"{B['tool_baseline_eagle3']['tpot_ms'] / B['tool_grammar_masked']['tpot_ms']:.3f}x, "
            f"grammar vs no_spec "
            f"{B['tool_no_spec']['tpot_ms'] / B['tool_grammar_masked']['tpot_ms']:.3f}x, "
            f"baseline vs no_spec "
            f"{B['tool_no_spec']['tpot_ms'] / B['tool_baseline_eagle3']['tpot_ms']:.3f}x",
            flush=True,
        )

    out_path = "/root/grammar-spec/robustness_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved {out_path}", flush=True)


if __name__ == "__main__":
    main()
