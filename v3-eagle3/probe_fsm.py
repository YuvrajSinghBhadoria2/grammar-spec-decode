"""Diagnose tool-schema FSM desync.

Probe A (no engine): regenerate the 8 tool-schema outputs (no_spec),
feed each output's tokens through a STANDALONE xgrammar matcher, and
report accept/reject statistics + first rejection context.
If standalone accepts ~100%, the in-engine bookkeeping is at fault.
"""

from __future__ import annotations

import json
import sys

sys.path.insert(0, "/root/grammar-spec")
from robustness import TOOL_PROMPTS, TOOL_SCHEMA  # noqa: E402
from eagle3_grammar_b1 import TOKENIZER_NAME  # noqa: E402

import xgrammar as xgr  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402


def standalone_accepts(tok, text: str) -> dict:
    ti = xgr.TokenizerInfo.from_huggingface(tok)
    cg = xgr.GrammarCompiler(ti).compile_grammar(
        xgr.Grammar.from_json_schema(json.dumps(TOOL_SCHEMA))
    )
    m = xgr.GrammarMatcher(cg, max_rollback_tokens=6)
    ids = tok.encode(text, add_special_tokens=False)
    n_ok = 0
    first_reject = None
    for idx, t in enumerate(ids):
        if m.is_terminated():
            break
        if m.accept_token(t):
            n_ok += 1
        else:
            first_reject = (idx, t, tok.decode(ids[max(0, idx - 8): idx + 1]))
            break
    return {
        "n_tokens": len(ids),
        "accepted": n_ok,
        "terminated": m.is_terminated(),
        "first_reject": first_reject,
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

    print("=== PROBE A: standalone matcher on real outputs ===", flush=True)
    for i, o in enumerate(outs):
        text = o.outputs[0].text
        ntok = len(o.outputs[0].token_ids)
        r = standalone_accepts(tok, text)
        rej = r["first_reject"]
        rej_s = f" reject@{rej[0]} tok={rej[1]} ctx={rej[2]!r}" if rej else ""
        print(
            f"PROMPT {i}: gen_tok={ntok} standalone_accept={r['accepted']}/"
            f"{r['n_tokens']} terminated={r['terminated']}{rej_s}",
            flush=True,
        )

    print("=== PROBE A2: matcher fed GENERATED token_ids (not re-encoded) ===", flush=True)
    for i, o in enumerate(outs):
        ids = list(o.outputs[0].token_ids)
        ti = xgr.TokenizerInfo.from_huggingface(tok)
        cg = xgr.GrammarCompiler(ti).compile_grammar(
            xgr.Grammar.from_json_schema(json.dumps(TOOL_SCHEMA))
        )
        m = xgr.GrammarMatcher(cg, max_rollback_tokens=6)
        n_ok = 0
        first_reject = None
        for idx, t in enumerate(ids):
            if m.is_terminated():
                break
            if m.accept_token(t):
                n_ok += 1
            else:
                first_reject = (idx, t, tok.decode(ids[max(0, idx - 8): idx + 1]))
                break
        rej_s = (
            f" reject@{first_reject[0]} tok={first_reject[1]} ctx={first_reject[2]!r}"
            if first_reject
            else ""
        )
        print(
            f"PROMPT {i}: accept={n_ok}/{len(ids)} "
            f"finish={o.outputs[0].finish_reason}{rej_s}",
            flush=True,
        )
    print("PROBE_A_DONE", flush=True)


if __name__ == "__main__":
    main()
