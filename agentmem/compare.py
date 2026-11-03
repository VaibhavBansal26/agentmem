"""Does the final pipeline beat the best baseline on the SAME questions? Plan and criterion: results/compare_PLAN.md.

    python -m agentmem.compare run --dataset longmemeval     # resumable, one JSON row per question
    python -m agentmem.compare run --dataset locomo
    python -m agentmem.compare report                        # paired bootstrap CIs and the verdict

Systems, all retrieval-only at k=10 over the full haystack of each question:
  ce_baseline  dense top-50 (bge-small) reranked by bge-reranker-base: the best baseline from Stage 1
  hybrid_ce    dense + BM25 + cross-encoder blend, no heuristics, no chaining
  pipeline     the final configuration (MemoryService defaults: + cue-gated recency bonus and entity chaining)
  dense, bm25  reference
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np

from . import loaders, metrics
from .retrievers import BM25, Dense
from .service import CHAIN, WEIGHTS, MemoryService

K = 10
OUT = Path("results/compare")
HYBRID_WEIGHTS = {**WEIGHTS, "scope": 0.0, "entity": 0.0, "recency": 0.0, "recency_cue": 0.0}
HYBRID_CHAIN = {**CHAIN, "w_upd": 0.0, "w_upd_cue": 0.0, "hops": 0}
SYSTEMS = ("pipeline", "hybrid_ce", "ce_baseline", "dense", "bm25")
DATA = {"longmemeval": (loaders.load_longmemeval, "data/longmemeval/longmemeval_s_cleaned.json"),
        "locomo": (loaders.load_locomo, "data/locomo/locomo10.json")}


class CEMemo:
    """One cross-encoder shared by every system, with a pair-level cache so a pair is scored once."""

    def __init__(self):
        from sentence_transformers import CrossEncoder

        self.ce, self.memo = CrossEncoder("BAAI/bge-reranker-base"), {}

    def __call__(self, pairs):
        pairs = [tuple(p) for p in pairs]
        todo = [p for p in dict.fromkeys(pairs) if p not in self.memo]
        if todo:
            self.memo.update(zip(todo, self.ce.predict(todo, batch_size=32, show_progress_bar=False)))
        return [float(self.memo[p]) for p in pairs]


def score_ids(ids, gold):
    return [metrics.recall_at_k(ids, gold, K), metrics.mrr(ids, gold)]


def run(dataset, limit=0):
    load, path = DATA[dataset]
    everything = load(path)  # loaded once: the LongMemEval file is large
    order = {q.id: i for i, q in enumerate(everything)}  # loader order, so the even/odd split matches Stage 2
    questions = [q for q in everything if q.gold_ids]  # abstention has no gold: nothing to score
    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / f"{dataset}.jsonl"
    done = {json.loads(l)["id"] for l in open(out_path)} if out_path.exists() else set()
    dense, bm25, ce = Dense(), BM25(), CEMemo()
    services = {}
    for n, q in enumerate(questions[: limit or None]):
        if q.id in done:
            continue
        hid = id(q.memories)
        if hid not in services:
            services.clear() if dataset == "longmemeval" else None  # LME: one haystack per question; LoCoMo: shared
            pair = []
            for w, c in ((WEIGHTS, CHAIN), (HYBRID_WEIGHTS, HYBRID_CHAIN)):
                s = MemoryService(now=q.asked_at, rerank=True, use_ce=False, dense=dense, bm25=bm25, scorer=ce,
                                  weights=w, chain_params=c)
                s.bulk_load(q.memories)
                pair.append(s)
            services[hid] = pair
        full, hybrid = services[hid]
        row = {"id": q.id, "qtype": q.qtype, "index": order[q.id], "systems": {}}
        row["systems"]["pipeline"] = score_ids([x["id"] for x in full.recall(q.text, k=K)], q.gold_ids)
        row["systems"]["hybrid_ce"] = score_ids([x["id"] for x in hybrid.recall(q.text, k=K)], q.gold_ids)
        cands = dense.retrieve(q.text, q.memories, 50, now=q.asked_at)
        s = np.asarray(ce([(q.text, m.text) for m in cands]))
        row["systems"]["ce_baseline"] = score_ids([cands[i].id for i in np.argsort(-s)[:K]], q.gold_ids)
        row["systems"]["dense"] = score_ids([m.id for m in dense.retrieve(q.text, q.memories, K, now=q.asked_at)], q.gold_ids)
        row["systems"]["bm25"] = score_ids([m.id for m in bm25.retrieve(q.text, q.memories, K, now=q.asked_at)], q.gold_ids)
        with open(out_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        if n % 25 == 0:
            print(f"{dataset} {n}/{len(questions)}", flush=True)


# --- analysis -------------------------------------------------------------------------------------
def load_rows(dataset):
    p = OUT / f"{dataset}.jsonl"
    return [json.loads(l) for l in open(p)] if p.exists() else []


def vec(rows, system, m):
    return np.array([r["systems"][system][m] for r in rows], dtype=float)


def paired_ci(a, b, n_boot=2000, seed=0):
    """Mean of (a - b) with a 95% paired bootstrap CI over questions."""
    d = a - b
    rng = random.Random(seed)
    idx = list(range(len(d)))
    boots = sorted(float(np.mean(d[[rng.choice(idx) for _ in idx]])) for _ in range(n_boot))
    return float(d.mean()), boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot) - 1]


def fmt(t):
    return f"{t[0]:+.3f} [{t[1]:+.3f}, {t[2]:+.3f}]"


def section(title, rows):
    out = [f"\n## {title} (n={len(rows)})\n", "| system | recall@10 | MRR |", "|---|---|---|"]
    for s in SYSTEMS:
        out.append(f"| {s} | {vec(rows, s, 0).mean():.3f} | {vec(rows, s, 1).mean():.3f} |")
    out += ["", "Paired difference (first minus second), mean [95% CI]:", "",
            "| comparison | recall@10 | MRR |", "|---|---|---|"]
    for a, b in (("pipeline", "ce_baseline"), ("hybrid_ce", "ce_baseline"), ("pipeline", "hybrid_ce")):
        out.append(f"| {a} - {b} | {fmt(paired_ci(vec(rows, a, 0), vec(rows, b, 0)))} | "
                   f"{fmt(paired_ci(vec(rows, a, 1), vec(rows, b, 1)))} |")
    types = sorted({r["qtype"] for r in rows})
    out += ["", "Per type, pipeline minus ce_baseline:", "", "| type | n | recall@10 | MRR |", "|---|---|---|---|"]
    wins = 0
    for t in types:
        sub = [r for r in rows if r["qtype"] == t]
        rc, mr = (paired_ci(vec(sub, "pipeline", m), vec(sub, "ce_baseline", m)) for m in (0, 1))
        wins += (rc[1] > 0) or (mr[1] > 0)
        out.append(f"| {t} | {len(sub)} | {fmt(rc)} | {fmt(mr)} |")
    return out, wins


def report():
    lme, loc = load_rows("longmemeval"), load_rows("locomo")
    dev = [r for r in lme if r["index"] % 2 == 1]  # the half the Stage 2 weights were NOT fit on
    out = ["# Does the final pipeline beat the best baseline on the same questions?\n",
           "Plan and criterion were written before running: `results/compare_PLAN.md`. Retrieval-only (recall@10, MRR).\n"]
    parts = {}
    for key, title, rows in (("dev", "LongMemEval_s, odd-indexed half (PRIMARY: weights were fit on the even half)", dev),
                             ("lme", "LongMemEval_s, all questions (reference only: includes the tuning half)", lme),
                             ("loc", "LoCoMo, all questions (held out from the Stage 2 weight fit)", loc)):
        if rows:
            lines, wins = section(title, rows)
            out += lines
            parts[key] = (rows, wins)
    verdict = ["", "## Verdict against the pre-registered criterion", ""]
    if "dev" in parts and "loc" in parts:
        rows, wins = parts["dev"]
        a_overall = any(paired_ci(vec(rows, "pipeline", m), vec(rows, "ce_baseline", m))[1] > 0 for m in (0, 1))
        a_types = wins >= 2
        lrows = parts["loc"][0]
        regress = any(paired_ci(vec(lrows, "pipeline", m), vec(lrows, "ce_baseline", m))[2] < 0 for m in (0, 1))
        verdict += [f"- LongMemEval_s (odd half): beats the baseline overall (95% CI above 0 on recall@10 or MRR): **{a_overall}**",
                    f"- LongMemEval_s (odd half): beats it in at least 2 question types (CI above 0 on either metric): **{a_types}** ({wins} types)",
                    f"- LoCoMo: significant regression (CI entirely below 0 on either metric): **{regress}**",
                    "", f"**Criterion met: {(a_overall or a_types) and not regress}**"]
    out += verdict
    text = "\n".join(out) + "\n"
    (OUT / "report.md").write_text(text)
    return text


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--dataset", choices=sorted(DATA), required=True)
    r.add_argument("--limit", type=int, default=0)
    sub.add_parser("report")
    a = ap.parse_args()
    if a.cmd == "run":
        run(a.dataset, a.limit)
    else:
        print(report())
