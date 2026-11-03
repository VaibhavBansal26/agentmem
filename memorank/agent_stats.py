"""Paired off-vs-on statistics for the agent runs: python -m memorank.agent_stats

Re-grades the stored raw answers with the current grader, so grader changes need no reruns.
"""
import json
import random
from collections import Counter
from pathlib import Path

from .agents_common import grade, summarize


def load(name):
    return {r["id"]: r for r in map(json.loads, open(f"results/agent_runs/{name}.jsonl"))}


def report():
    runs = {n: load(n) for n in ("anthropic_baseline", "anthropic_reranker",
                                 "openai_baseline", "openai_reranker")}
    ok = {n: {i: grade(r["answer"], r["gold"], r["qtype"]) for i, r in d.items()} for n, d in runs.items()}
    qtype = {i: r["qtype"] for i, r in runs["anthropic_baseline"].items()}
    ids, types = sorted(qtype), sorted(set(qtype.values()))
    out = [summarize([f"results/agent_runs/{n}.jsonl" for n in runs]), "",
           "## Paired comparison, reranker off (A) vs on (B)", "",
           "Each question is asked in both variants; the difference is bootstrapped over questions "
           "(2000 resamples, 95% CI). Single run per cell, so model nondeterminism is not averaged out.", ""]
    for agent in ("anthropic", "openai"):
        a, b = ok[f"{agent}_baseline"], ok[f"{agent}_reranker"]
        d = [int(b[i]) - int(a[i]) for i in ids]
        rng = random.Random(0)
        boots = sorted(sum(rng.choices(d, k=len(d))) / len(d) for _ in range(2000))
        out += [f"### {agent}", "",
                f"Accuracy {sum(a.values()) / len(a):.3f} -> {sum(b.values()) / len(b):.3f}; "
                f"paired difference {sum(d) / len(d):+.3f} [{boots[50]:+.3f}, {boots[1949]:+.3f}]; "
                f"fixed by reranker {sum(x == 1 for x in d)}, broken by reranker {sum(x == -1 for x in d)}.", "",
                "| type | n | off | on |", "|---|---|---|---|"]
        for t in types:
            s = [i for i in ids if qtype[i] == t]
            out.append(f"| {t} | {len(s)} | {sum(a[i] for i in s) / len(s):.2f} | {sum(b[i] for i in s) / len(s):.2f} |")
        out.append("")
    miss = {n: Counter(r["qtype"] for r in d.values() if not ok[n][r["id"]]) for n, d in runs.items()}
    out += ["## Remaining misses by type", ""] + [f"- {n}: {dict(c)}" for n, c in miss.items()]
    text = "\n".join(out) + "\n"
    Path("results/stage4_agents.md").write_text(text)
    return text


if __name__ == "__main__":
    print(report())
