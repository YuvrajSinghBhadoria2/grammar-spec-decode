"""Tool-schema experiment rerun — FIXED schema clobbering.

Imports only eagle3_grammar_fresh (no robustness/eg — their module-level
schema writes clobber SCHEMA_PATH when the engine worker re-imports).
Writes the tool schema to SCHEMA_PATH in the parent right before each
engine start. Warmup is grammar-constrained.
"""

from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, "/root/grammar-spec")
import eagle3_grammar_fresh as egf  # noqa: E402  (fresh patch, no schema write)

from eagle3_grammar_fresh import (  # noqa: E402
    BASE_STATS_PATH,
    DIAG_PATH,
    ENABLE_PATH,
    SCHEMA_PATH,
    STATS_PATH,
    _set_flag,
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


def run_one(name: str, mode: str, spec: dict | None) -> dict:
    use_spec = mode != "none"
    with open(SCHEMA_PATH, "w") as f:
        json.dump(TOOL_SCHEMA, f)
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
        structured_outputs=StructuredOutputsParams(json=TOOL_SCHEMA),
    )
    if use_spec:
        llm.generate(
            [TOOL_PROMPTS[0]],
            SamplingParams(
                temperature=0.0,
                max_tokens=32,
                structured_outputs=StructuredOutputsParams(json=TOOL_SCHEMA),
            ),
        )

    sys.stdout.flush()
    start = time.perf_counter()
    outputs = llm.generate(TOOL_PROMPTS, sp)
    elapsed = time.perf_counter() - start
    sys.stdout.flush()

    valid = 0
    total_tokens = 0
    for out in outputs:
        total_tokens += len(out.outputs[0].token_ids)
        if valid_tool(out.outputs[0].text):
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
    st = result.get("stats", {})
    tpp = (
        total_tokens / st["propose_calls"] if st.get("propose_calls") else None
    )
    print(
        f"  [{name}] Valid={valid}/{n}  TPOT={result['tpot_ms']:.2f}ms  "
        f"AvgTok={total_tokens / n:.1f}  proposes={st.get('propose_calls', '-')}  "
        f"tok/propose={tpp and round(tpp, 3)}  "
        f"acc_cmt={st.get('accepted_committed')} rej_cmt={st.get('rejected_committed')}",
        flush=True,
    )

    del llm
    import torch

    torch.cuda.empty_cache()
    return result


def main() -> None:
    open("/root/grammar-spec/fresh_fail.log", "w").close()
    results = {}
    results["tool_no_spec_fixed"] = run_one("tool_no_spec_fixed", "none", None)
    results["tool_baseline_fixed"] = run_one(
        "tool_baseline_fixed",
        "eagle3",
        {
            "method": "eagle3",
            "model": "ruipeterpan/Qwen2.5-7B-Instruct_EAGLE3_UltraChat",
            "num_speculative_tokens": 5,
        },
    )
    results["tool_grammar_fixed"] = run_one(
        "tool_grammar_fixed",
        "eagle3_grammar",
        {
            "method": "eagle3",
            "model": "ruipeterpan/Qwen2.5-7B-Instruct_EAGLE3_UltraChat",
            "num_speculative_tokens": 5,
        },
    )

    _set_flag(ENABLE_PATH, False)
    _set_flag(DIAG_PATH, False)

    print(flush=True)
    print("=" * 72, flush=True)
    print("TOOL-SCHEMA FIXED RERUN", flush=True)
    print("=" * 72, flush=True)
    ns, base, gram = (
        results["tool_no_spec_fixed"],
        results["tool_baseline_fixed"],
        results["tool_grammar_fixed"],
    )
    print(
        f"no_spec  Valid={ns['valid']}/{ns['total']} TPOT={ns['tpot_ms']:.2f}",
        flush=True,
    )
    print(
        f"baseline Valid={base['valid']}/{base['total']} TPOT={base['tpot_ms']:.2f} "
        f"({ns['tpot_ms'] / base['tpot_ms']:.3f}x vs no_spec)",
        flush=True,
    )
    print(
        f"grammar  Valid={gram['valid']}/{gram['total']} TPOT={gram['tpot_ms']:.2f} "
        f"({ns['tpot_ms'] / gram['tpot_ms']:.3f}x vs no_spec, "
        f"{base['tpot_ms'] / gram['tpot_ms']:.3f}x vs baseline)",
        flush=True,
    )

    with open("/root/grammar-spec/tool_fixed_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("Saved /root/grammar-spec/tool_fixed_results.json", flush=True)


if __name__ == "__main__":
    main()
