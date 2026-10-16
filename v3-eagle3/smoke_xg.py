"""Smoke test: Qwen tokenizer + xgrammar + agentic schema end-to-end."""

import json

import torch
import xgrammar as xgr
from transformers import AutoTokenizer

SCHEMA = {
    "type": "object",
    "properties": {
        "tool": {"type": "string"},
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

tok = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True
)
ti = xgr.TokenizerInfo.from_huggingface(tok)
print("xgrammar vocab_size:", ti.vocab_size)

compiler = xgr.GrammarCompiler(ti)
cg = compiler.compile_grammar(xgr.Grammar.from_json_schema(json.dumps(SCHEMA)))
print("compiled ok:", type(cg).__name__)

m = xgr.GrammarMatcher(cg, max_rollback_tokens=8)
print("matcher ok")

shape = xgr.get_bitmask_shape(1, ti.vocab_size)
print("bitmask shape:", shape)
bm = xgr.allocate_token_bitmask(1, ti.vocab_size)
ok = m.fill_next_token_bitmask(bm, 0)
print("fill ok:", ok, "nonzero bits:", int((bm != 0).sum()))

# Simulate target-vocab logits (Qwen2.5-7B padded vocab = 152064)
logits = torch.zeros(1, 152064)
xgr.apply_token_bitmask_inplace(logits, bm, indices=[0])
valid = int((logits > float("-inf")).sum())
print("valid after mask:", valid)

# Walk the grammar: '{' then '"tool"' etc.
# Find token ids for a few JSON tokens
candidates = ["{", '"tool"', ":", '"', "weather", '"', "}"]
for s in candidates:
    tid = tok.convert_tokens_to_ids(s) if s in tok.get_vocab() else None
    if tid is None:
        ids = tok.encode(s, add_special_tokens=False)
        tid = ids[0] if ids else None
    print(f"  token {s!r} -> id {tid}")
    if tid is not None:
        accepted = m.accept_token(int(tid))
        print(f"    accept_token: {accepted}, terminated: {m.is_terminated()}")

# Test rollback
m2 = xgr.GrammarMatcher(cg, max_rollback_tokens=8)
before = m2.accept_token(198)  # '{' likely
print("rollback test: accept 198 ->", before)
m2.rollback(1)
print("after rollback terminated:", m2.is_terminated())
print("ALL OK")
