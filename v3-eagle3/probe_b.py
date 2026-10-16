"""Probe B: instrumented tool-grammar run — counters for silent failure
paths in the eagle3_grammar_b1 patch (committed=None, rollback failures,
prep path used)."""

from __future__ import annotations

import collections
import json
import sys

sys.path.insert(0, "/root/grammar-spec")
import eagle3_grammar_b1 as eg  # noqa: E402  (installs the patch)

from vllm.v1.spec_decode.eagle import EagleProposer  # noqa: E402

COUNTERS = collections.Counter()
COMMITTED_LEN = collections.Counter()

_padded = EagleProposer.prepare_next_token_ids_padded
_cpu = EagleProposer.prepare_next_token_ids_cpu


def wrapped_padded(self, *a, **kw):
    COUNTERS["prep_padded_calls"] += 1
    try:
        out = _padded(self, *a, **kw)
        if self._g_committed is None:
            COUNTERS["prep_padded_committed_none"] += 1
        else:
            COMMITTED_LEN["padded_" + ",".join(str(len(c)) for c in self._g_committed)] += 1
        return out
    except Exception:
        COUNTERS["prep_padded_outer_exc"] += 1
        raise


def wrapped_cpu(self, *a, **kw):
    COUNTERS["prep_cpu_calls"] += 1
    try:
        out = _cpu(self, *a, **kw)
        if self._g_committed is None:
            COUNTERS["prep_cpu_committed_none"] += 1
        else:
            COMMITTED_LEN["cpu_" + ",".join(str(len(c)) for c in self._g_committed)] += 1
        return out
    except Exception:
        COUNTERS["prep_cpu_outer_exc"] += 1
        raise


EagleProposer.prepare_next_token_ids_padded = wrapped_padded
EagleProposer.prepare_next_token_ids_cpu = wrapped_cpu


def main() -> None:
    from robustness import TOOL_PROMPTS, TOOL_SCHEMA, run_one, spec_cfg

    r = run_one(
        "tool_grammar_diag",
        "eagle3_grammar",
        spec_cfg(5),
        TOOL_PROMPTS,
        TOOL_SCHEMA,
        lambda t: True,
    )
    print("COUNTERS:", json.dumps(COUNTERS), flush=True)
    print("COMMITTED_LEN:", json.dumps(COMMITTED_LEN), flush=True)
    print("stats:", json.dumps(r.get("stats", {})), flush=True)


if __name__ == "__main__":
    main()
