"""Identify which of the 12 prompts fails JSON validity (temp=0, no spec)."""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eagle3_grammar_b1 import AGENTIC_PROMPTS, AGENTIC_SCHEMA  # noqa: E402

from vllm import LLM, SamplingParams  # noqa: E402
from vllm.sampling_params import StructuredOutputsParams  # noqa: E402

def main():
    global llm
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
        structured_outputs=StructuredOutputsParams(json=AGENTIC_SCHEMA),
    )
    outs = llm.generate(AGENTIC_PROMPTS, sp)
    for i, o in enumerate(outs):
        text = o.outputs[0].text
        status = "OK"
        detail = ""
        try:
            parsed = json.loads(text)
            missing = [k for k in AGENTIC_SCHEMA["required"] if k not in parsed]
            if missing:
                status = "MISSING_KEYS"
                detail = str(missing)
            else:
                status = "VALID"
        except json.JSONDecodeError as e:
            status = "JSON_ERROR"
            detail = str(e)
        ntok = len(o.outputs[0].token_ids)
        if status != "VALID":
            print(
                f"PROMPT {i}: {status} {detail}  ntok={ntok} "
                f"finish={o.outputs[0].finish_reason}",
                flush=True,
            )
            print(f"  text[:300]={text[:300]!r}", flush=True)
            print(f"  text[-300:]={text[-300:]!r}", flush=True)
        else:
            print(f"PROMPT {i}: VALID ntok={ntok}", flush=True)
    print("PROBE_DONE", flush=True)


if __name__ == "__main__":
    main()
