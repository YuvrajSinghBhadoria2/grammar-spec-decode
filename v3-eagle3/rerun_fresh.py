"""Selective v4 rerun: fresh-matcher grammar patch on (a) tool schema,
(b) weather schema, plus a same-session weather baseline for comparison.

Import order matters: eagle3_grammar_fresh installs its patch FIRST;
robustness's import of eagle3_grammar_b1 then skips (patch flag already set).
"""

from __future__ import annotations

import json
import sys

sys.path.insert(0, "/root/grammar-spec")

import eagle3_grammar_fresh as egf  # noqa: E402  (installs FRESH patch first)

from robustness import (  # noqa: E402
    TOOL_PROMPTS,
    TOOL_SCHEMA,
    run_one,
    spec_cfg,
    valid_tool,
    valid_weather,
)
from eagle3_grammar_fresh import AGENTIC_PROMPTS, AGENTIC_SCHEMA  # noqa: E402


def main() -> None:
    results = {}
    results["tool_grammar_fresh"] = run_one(
        "tool_grammar_fresh", "eagle3_grammar", spec_cfg(5),
        TOOL_PROMPTS, TOOL_SCHEMA, valid_tool,
    )
    results["weather_grammar_fresh"] = run_one(
        "weather_grammar_fresh", "eagle3_grammar", spec_cfg(5),
        AGENTIC_PROMPTS, AGENTIC_SCHEMA, valid_weather,
    )
    results["weather_baseline_session2"] = run_one(
        "weather_baseline_session2", "eagle3", spec_cfg(5),
        AGENTIC_PROMPTS, AGENTIC_SCHEMA, valid_weather,
    )

    print(flush=True)
    print("=" * 72, flush=True)
    print("V4 FRESH-MATCHER RERUN SUMMARY", flush=True)
    print("=" * 72, flush=True)
    for name, r in results.items():
        st = r.get("stats", {})
        tpp = (
            r["total_tokens"] / st["propose_calls"]
            if st.get("propose_calls")
            else None
        )
        print(
            f"{name:<28} Valid={r['valid']}/{r['total']}  "
            f"TPOT={r['tpot_ms']:.2f}ms  tok/propose={tpp and round(tpp, 3)}  "
            f"acc_cmt={st.get('accepted_committed')} "
            f"rej_cmt={st.get('rejected_committed')} "
            f"fallback={st.get('fallback_first_valid')}",
            flush=True,
        )
    tg = results["tool_grammar_fresh"]
    wb = results["weather_baseline_session2"]
    wg = results["weather_grammar_fresh"]
    print(
        f"tool fresh vs v3 tool baseline 13.42: "
        f"{13.42 / tg['tpot_ms']:.3f}x; vs tool no_spec 10.96: "
        f"{10.96 / tg['tpot_ms']:.3f}x",
        flush=True,
    )
    print(
        f"weather fresh vs session2 baseline: {wb['tpot_ms'] / wg['tpot_ms']:.3f}x",
        flush=True,
    )

    with open("/root/grammar-spec/fresh_rerun_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("Saved /root/grammar-spec/fresh_rerun_results.json", flush=True)


if __name__ == "__main__":
    main()
