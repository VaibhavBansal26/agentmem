"""Run the domain questions through an agent with the reranker off or on.

    python -m agentmem.run_agents --agent anthropic --reranker on
    python -m agentmem.run_agents --agent openai --reranker off
    python -m agentmem.run_agents --summarize

Needs ANTHROPIC_API_KEY / OPENAI_API_KEY, exported in your shell or put in a git-ignored .env
(see .env.example). Every tool call is logged to
results/agent_runs/<agent>_<variant>.jsonl and the table is built from those logs.
"""
import argparse
import asyncio
import glob
import os
from pathlib import Path

from .agents_common import grade, log_row, summarize
from .loaders import load_domain


def load_env(path=".env"):
    """Read KEY=VALUE lines from a git-ignored .env; existing environment variables win."""
    if Path(path).exists():
        for line in Path(path).read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("\"'"))


async def main():
    load_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", choices=["anthropic", "openai"])
    ap.add_argument("--reranker", choices=["on", "off"], default="on")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args()
    if a.summarize:
        print(summarize(sorted(glob.glob("results/agent_runs/*.jsonl"))))
        return
    if not a.agent:
        ap.error("--agent is required unless --summarize")
    if a.agent == "anthropic":
        from .agent_anthropic import run
    else:
        from .agent_openai import run
    questions = load_domain("datasets/securitydomain", a.limit)
    path = f"results/agent_runs/{a.agent}_{'reranker' if a.reranker == 'on' else 'baseline'}.jsonl"
    streak = 0
    async for q, text, calls in run(questions, a.reranker == "on"):
        errs = sum(bool(c.get("error")) for c in calls)
        log_row(path, {"id": q.id, "qtype": q.qtype, "question": q.text, "answer": text, "gold": q.answer,
                       "correct": grade(text, q.answer, q.qtype), "calls": calls, "tool_errors": errs})
        print(q.id, q.qtype, "ok" if grade(text, q.answer, q.qtype) else "MISS", len(calls), "calls",
              f"TOOL ERRORS: {errs}" if errs else "", flush=True)
        streak = streak + 1 if errs else 0
        if streak >= 3:  # a broken tool would otherwise be scored as model mistakes
            raise SystemExit("Aborting: 3 consecutive questions had tool errors. Fix the server first.")


if __name__ == "__main__":
    asyncio.run(main())
