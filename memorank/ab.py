"""Online A/B over synthetic sessions: a flag assigns each session to reranker A (dense top-k) or B.

    python -m memorank.ab --sessions 300 --db ab/sessions.db

Resumable: sessions already in the DB are skipped. Plan and decision rule: ab/PREREG.md.
"""
import argparse
import asyncio
import random
import time

from agents import Runner

from . import agent_openai
from .agents_common import grade
from .flags import Flags
from .loaders import load_domain
from .run_agents import load_env
from .telemetry import sinks

MODEL = agent_openai.MODEL


def usage(res):
    i = o = 0
    for r in getattr(res, "raw_responses", []) or []:
        u = getattr(r, "usage", None)
        i += getattr(u, "input_tokens", 0) or 0
        o += getattr(u, "output_tokens", 0) or 0
    return i, o


async def run(n, db, seed=0):
    load_env()
    flags, out = Flags(db), sinks(db)
    flags.set("reranker", {"A": 0.5, "B": 0.5})
    questions, rnd, done = load_domain("datasets/securitydomain"), random.Random(seed), out[0].done()
    agents = {v: agent_openai.build(v == "B") for v in ("A", "B")}
    streak = 0
    async with agents["A"][1], agents["A"][2], agents["B"][1], agents["B"][2]:
        for i in range(n):
            q = rnd.choice(questions)  # drawn before the skip so a resumed run reproduces the sequence
            sid = f"s{i:04d}"
            if sid in done:
                continue
            variant = flags.variant("reranker", sid)
            t = time.time()
            res = await Runner.run(agents[variant][0], q.text, max_turns=10)
            calls = agent_openai.tool_calls(res)
            errs = sum(bool(c["error"]) for c in calls)
            tin, tout = usage(res)
            text = str(res.final_output)
            row = dict(session_id=sid, variant=variant, question_id=q.id, qtype=q.qtype,
                       correct=int(grade(text, q.answer, q.qtype)), tool_calls=len(calls), tool_errors=errs,
                       input_tokens=tin, output_tokens=tout, latency_s=round(time.time() - t, 2),
                       model=MODEL, answer=text)
            for s in out:
                s.record(**row)
            print(sid, variant, q.id, "ok" if row["correct"] else "MISS", f"errors={errs}", flush=True)
            streak = streak + 1 if errs else 0
            if streak >= 3:
                raise SystemExit("Aborting: 3 consecutive sessions had tool errors. Fix the server first.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", type=int, default=300)
    ap.add_argument("--db", default="ab/sessions.db")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    import os

    os.makedirs(os.path.dirname(a.db) or ".", exist_ok=True)
    asyncio.run(run(a.sessions, a.db, a.seed))
