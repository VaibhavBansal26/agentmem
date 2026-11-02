"""Stage 2 spike: `featurize` caches candidate features, `tune` fits weights and reports.

Rule: weights are fit on even-indexed LongMemEval_s questions only. Odd-indexed
questions are the dev check; LoCoMo is scored once with the frozen weights.
"""
import argparse
import pickle
from pathlib import Path

import numpy as np

from . import loaders, metrics
from .rerank import WEIGHTS, BM25, Dense, featurize, rank_ids

FEATS = Path(".cache/features")
GRID = {
    "bm25": [0, 0.25, 0.5, 1, 2],
    "scope": [0, 0.5, 1, 2, 4],
    "entity": [0, 0.5, 1, 2],
    "recency": [0, 0.25, 0.5, 1],
    "recency_cue": [0, 0.5, 1, 2, 4],
}
K = 10


def cmd_featurize(a):
    qs = {"longmemeval": loaders.load_longmemeval, "locomo": loaders.load_locomo}[a.dataset](
        a.path, a.limit
    )
    dense, bm25 = Dense(), BM25()
    out = []
    for i, q in enumerate(qs):
        out.append(featurize(q, dense, bm25))
        if i % 50 == 0:
            print(f"{a.dataset} {i}/{len(qs)}", flush=True)
    FEATS.mkdir(parents=True, exist_ok=True)
    pickle.dump(out, open(FEATS / f"{a.dataset}.pkl", "wb"))


def evaluate(feats, w):
    rows = []
    for F in feats:
        ids = rank_ids(F, w)[:K]
        rows.append((F["qtype"], metrics.recall_at_k(ids, F["gold"], K), metrics.mrr(ids, F["gold"])))
    return rows


def objective(rows):
    r = [x[1] for x in rows if x[1] is not None]
    m = [x[2] for x in rows if x[2] is not None]
    return (sum(r) + sum(m)) / (len(r) + len(m))


def tune(feats, rounds=3):
    w = dict(WEIGHTS)
    best = objective(evaluate(feats, w))
    for _ in range(rounds):
        for name, grid in GRID.items():
            for v in grid:
                trial = {**w, name: v}
                s = objective(evaluate(feats, trial))
                if s > best + 1e-6:
                    best, w = s, trial
    return w, best


def table(rows):
    by = {}
    for t, r, m in rows:
        for key in (t, "ALL"):
            by.setdefault(key, []).append((r, m))
    lines = ["| type | n | recall@10 | MRR |", "|---|---|---|---|"]
    for t in sorted(by, key=lambda t: (t == "ALL", t)):
        rs = metrics.mean_ci([x[0] for x in by[t]])
        ms = metrics.mean_ci([x[1] for x in by[t]])
        f = lambda v: "n/a" if v[0] is None else f"{v[0]:.3f} [{v[1]:.3f}, {v[2]:.3f}]"
        lines.append(f"| {t} | {len(by[t])} | {f(rs)} | {f(ms)} |")
    return "\n".join(lines)


def cmd_tune(a):
    lme = pickle.load(open(FEATS / "longmemeval.pkl", "rb"))
    loc = pickle.load(open(FEATS / "locomo.pkl", "rb"))
    fit, dev = lme[0::2], lme[1::2]
    # tune.py never sees `dev` or LoCoMo during fitting.
    w, fit_score = tune(fit)
    hybrid = {**WEIGHTS, "bm25": 1.0}
    out = [f"# Stage 2 spike (k={K}, retrieval only, no training)\n",
           f"Fit on {len(fit)} even-indexed LongMemEval_s questions; dev = {len(dev)} odd-indexed; "
           f"LoCoMo ({len(loc)} questions) scored once with frozen weights.\n",
           f"Tuned weights: `{w}` (fit objective {fit_score:.3f}, mean of recall@10 and MRR)\n"]
    for name, feats in (("LongMemEval_s dev half", dev), ("LoCoMo (held out)", loc)):
        out.append(f"\n## {name}\n")
        cfgs = [("dense only", dict(WEIGHTS)), ("dense+bm25 (w=1)", hybrid), ("tuned reranker", w)]
        cfgs += [(f"tuned minus {k}", {**w, k: 0.0}) for k in WEIGHTS if w[k] != 0]
        out.append("| config | recall@10 | MRR |\n|---|---|---|")
        for label, cw in cfgs:
            rows = evaluate(feats, cw)
            rr = metrics.mean_ci([x[1] for x in rows])
            mm = metrics.mean_ci([x[2] for x in rows])
            out.append(f"| {label} | {rr[0]:.3f} [{rr[1]:.3f}, {rr[2]:.3f}] | "
                       f"{mm[0]:.3f} [{mm[1]:.3f}, {mm[2]:.3f}] |")
        out.append(f"\nPer type, tuned reranker:\n\n{table(evaluate(feats, w))}")
    cand = [metrics.recall_at_k(F["ids"], F["gold"], len(F["ids"])) for F in lme]
    out.append(f"\nCandidate-pool recall ceiling on LongMemEval_s: {metrics.mean_ci(cand)[0]:.3f}")
    text = "\n".join(out) + "\n"
    print(text)
    Path("results").mkdir(exist_ok=True)
    Path("results/stage2_spike.md").write_text(text)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(required=True)
    f = sub.add_parser("featurize")
    f.add_argument("--dataset", choices=["longmemeval", "locomo"], required=True)
    f.add_argument("--path", required=True)
    f.add_argument("--limit", type=int, default=0)
    f.set_defaults(fn=cmd_featurize)
    t = sub.add_parser("tune")
    t.set_defaults(fn=cmd_tune)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
