# LinkedIn post — grammar-aware speculative decoding

> Copy-paste draft for LinkedIn. Attach the blog as a link card for the best preview.

---

Speculative decoding was slowing our agents down — not speeding them up.

I spent the last two weeks measuring why, and the answer surprised me:

Under JSON schema constraints (what every agent workload runs on), 84.9% of speculative drafts get rejected by the grammar FSM. Unmasked EAGLE3 draft heads ended up 0.97× slower than no speculation at all — a net loss.

So I built the fix: grammar-masked draft logits — apply the xgrammar bitmask before argmax in the draft loop, not just at verification.

Results on vLLM 0.11.2 (A100, Qwen2.5-7B):

→ 1.19× faster than unmasked EAGLE3 (weather tool schema)
→ 1.31× faster + +50% draft acceptance on an enum-heavy tool-call schema
→ Wins at every draft length (K = 3, 5, 7)
→ Bonus: restores output validity that plain spec-decode destroys (3/8 → 5/8)

Also found and root-caused an unsound rollback() in xgrammar 0.1.25 — fixed with a fresh-matcher-per-propose design.

Filed an upstream design issue → https://github.com/vllm-project/vllm/issues/62130

Full writeup with every benchmark, dead end, and raw data:
https://yuvrajsinghbhadoria2.github.io/grammar-spec-decoding/

Repo: https://github.com/YuvrajSinghBhadoria2/grammar-spec-decode
