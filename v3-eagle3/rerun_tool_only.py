import os
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

import json
import sys

sys.path.insert(0, _BASE_DIR)
import eagle3_grammar_fresh  # noqa: E402  (fresh patch)

from robustness import (  # noqa: E402
    TOOL_PROMPTS,
    TOOL_SCHEMA,
    run_one,
    spec_cfg,
    valid_tool,
)


def main() -> None:
    open(os.path.join(_BASE_DIR, "fresh_fail.log"), "w").close()
    r = run_one(
        "tool_grammar_fresh2", "eagle3_grammar", spec_cfg(5),
        TOOL_PROMPTS, TOOL_SCHEMA, valid_tool,
    )
    print("stats:", json.dumps(r.get("stats", {})), flush=True)


if __name__ == "__main__":
    main()
