"""Probe C: can an xgrammar matcher survive accept-4 + rollback-4 on real
tool-schema streams? Emulates the propose() draft/rollback cycle with the
REAL generated token stream (no engine needed after generation)."""


from __future__ import annotations
import os
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))


import json
import sys

sys.path.insert(0, _BASE_DIR)
from robustness import TOOL_PROMPTS, TOOL_SCHEMA  # noqa: E402
from eagle3_grammar_b1 import TOKENIZER_NAME  # noqa: E402

import xgrammar as xgr  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402


def emulate_spec_cycle(ti, cg, ids: list[int], window: int = 4) -> dict:
    """Replay stream; at each position, accept `window` upcoming tokens as
    drafts, roll them back, then commit them for real (as the target would)."""
    m = xgr.GrammarMatcher(cg, max_rollback_tokens=window + 1)
    n = len(ids)
    pos = 0
    rollbacks_ok = 0
    rollbacks_fail = 0
    post_rollback_reject = None
    while pos < n:
        if m.is_terminated():
            break
        drafts = ids[pos: pos + window]
        accepted_before = pos
        nacc = 0
        for t in drafts:
            if m.is_terminated():
                break
            if m.accept_token(t):
                nacc += 1
            else:
                break
        # roll back the drafts
        try:
            m.rollback(nacc)
            rollbacks_ok += 1
        except Exception as e:
            rollbacks_fail += 1
            post_rollback_reject = ("rollback_exc", pos, str(e))
            break
        # now commit drafts for real
        for j, t in enumerate(drafts[:nacc] if nacc else []):
            if m.is_terminated():
                break
            if not m.accept_token(t):
                post_rollback_reject = (
                    "commit_after_rollback", pos + j, t,
                    # what would a FRESH matcher at pos+j say?
                )
                break
            pos += 1
        else:
            if nacc == 0:
                # drafts all rejected (none valid) — commit one token directly
                if m.accept_token(ids[pos]):
                    pos += 1
                else:
                    post_rollback_reject = ("direct_commit", pos, ids[pos])
                    break
            continue
        break
    return {
        "consumed": pos,
        "total": n,
        "rollbacks_ok": rollbacks_ok,
        "rollbacks_fail": rollbacks_fail,
        "first_fail": post_rollback_reject,
        "terminated": m.is_terminated(),
    }


def main() -> None:
    tok = AutoTokenizer.from_pretrained(TOKENIZER_NAME, trust_remote_code=True)
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    llm = LLM(
        model="Qwen/Qwen2.5-7B-Instruct",
        max_model_len=4096,
        dtype="bfloat16",
        max_num_seqs=1,
        gpu_memory_utilization=0.9,
        disable_log_stats=True,
    )
    sp = SamplingParams(
        temperature=0.0,
        max_tokens=512,
        structured_outputs=StructuredOutputsParams(json=TOOL_SCHEMA),
    )
    outs = llm.generate(TOOL_PROMPTS, sp)

    ti = xgr.TokenizerInfo.from_huggingface(tok)
    cg = xgr.GrammarCompiler(ti).compile_grammar(
        xgr.Grammar.from_json_schema(json.dumps(TOOL_SCHEMA))
    )
    print("=== PROBE C: accept-window + rollback cycle on real streams ===", flush=True)
    for i, o in enumerate(outs):
        ids = list(o.outputs[0].token_ids)
        r = emulate_spec_cycle(ti, cg, ids)
        print(
            f"PROMPT {i}: consumed={r['consumed']}/{r['total']} "
            f"rollbacks_ok={r['rollbacks_ok']} fail={r['rollbacks_fail']} "
            f"first_fail={r['first_fail']}",
            flush=True,
        )

    # Control: same emulation on a WEATHER-schema stream (known-good schema)
    print("=== PROBE C control: weather schema stream ===", flush=True)
    sys.path.insert(0, _BASE_DIR)
    from eagle3_grammar_b1 import AGENTIC_PROMPTS, AGENTIC_SCHEMA

    cg_w = xgr.GrammarCompiler(ti).compile_grammar(
        xgr.Grammar.from_json_schema(json.dumps(AGENTIC_SCHEMA))
    )
    outs_w = llm.generate(
        AGENTIC_PROMPTS,
        SamplingParams(
            temperature=0.0,
            max_tokens=256,
            structured_outputs=StructuredOutputsParams(json=AGENTIC_SCHEMA),
        ),
    )
    for i, o in enumerate(outs_w[:3]):
        ids = list(o.outputs[0].token_ids)
        r = emulate_spec_cycle(ti, cg_w, ids)
        print(
            f"W_PROMPT {i}: consumed={r['consumed']}/{r['total']} "
            f"rollbacks_ok={r['rollbacks_ok']} fail={r['rollbacks_fail']} "
            f"first_fail={r['first_fail']}",
            flush=True,
        )
    print("PROBE_C_DONE", flush=True)


if __name__ == "__main__":
    main()
