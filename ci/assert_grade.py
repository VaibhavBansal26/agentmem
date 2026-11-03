"""promptfoo assertion: the repo's own deterministic grader (stemmed clause words, exact numbers)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from memorank.agents_common import grade  # noqa: E402


def get_assert(output, context):
    v = context["vars"]
    ok = grade(str(output), v["gold"], v["qtype"])
    return {"pass": bool(ok), "score": 1.0 if ok else 0.0,
            "reason": f"gold={v['gold']!r} answer={str(output)[:140]!r}"}
