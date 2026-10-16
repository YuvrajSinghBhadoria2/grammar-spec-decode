"""Probe D: file-based logging from inside the engine worker —
prep path used, committed contents, matcher identity stability."""


from __future__ import annotations
import os
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))


import sys

sys.path.insert(0, _BASE_DIR)
import eagle3_grammar_b1 as eg  # noqa: E402

from vllm.v1.spec_decode.eagle import EagleProposer  # noqa: E402

DBG = os.path.join(_BASE_DIR, "dbg.log")
_padded = EagleProposer.prepare_next_token_ids_padded
_cpu = EagleProposer.prepare_next_token_ids_cpu
_propose = EagleProposer.propose
_calls = {"n": 0}


def _log(msg: str) -> None:
    with open(DBG, "a") as f:
        f.write(msg + "\n")


def wrapped_padded(self, *a, **kw):
    out = _padded(self, *a, **kw)
    if _calls["n"] < 8:
        _log(f"PREP padded committed={getattr(self, '_g_committed', 'MISSING')}")
    return out


def wrapped_cpu(self, *a, **kw):
    out = _cpu(self, *a, **kw)
    if _calls["n"] < 8:
        _log(f"PREP cpu committed={getattr(self, '_g_committed', 'MISSING')}")
    return out


def wrapped_propose(self, *a, **kw):
    _calls["n"] += 1
    n = _calls["n"]
    if n <= 8 or n % 500 == 0:
        rids = list(self.runner.input_batch.req_ids)
        _log(
            f"PROPOSE#{n} rids={rids} nmatchers={len(self._g_matchers)} "
            f"committed={self._g_committed} enabled={eg._grammar_enabled()}"
        )
    out = _propose(self, *a, **kw)
    if n <= 8:
        _log(
            f"  after#{n} stats_propose={self._g_stats['propose_calls']} "
            f"acc={self._g_stats['accepted_committed']} "
            f"rej={self._g_stats['rejected_committed']} "
            f"adv={self._g_stats['draft_advances']}"
        )
    return out


EagleProposer.prepare_next_token_ids_padded = wrapped_padded
EagleProposer.prepare_next_token_ids_cpu = wrapped_cpu
EagleProposer.propose = wrapped_propose


def main() -> None:
    from robustness import TOOL_PROMPTS, TOOL_SCHEMA, run_one, spec_cfg

    open(DBG, "w").close()
    r = run_one(
        "tool_grammar_dbg", "eagle3_grammar", spec_cfg(5),
        TOOL_PROMPTS, TOOL_SCHEMA, lambda t: True,
    )
    print("stats:", r.get("stats", {}), flush=True)
    print("DBG lines:", sum(1 for _ in open(DBG)), flush=True)


if __name__ == "__main__":
    main()
