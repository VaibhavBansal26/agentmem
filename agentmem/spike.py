"""Stage 2 spike: `featurize` caches candidate features, `tune` fits weights and reports.

Rule: weights are fit on even-indexed LongMemEval_s questions only. Odd-indexed
questions are the dev check; LoCoMo is scored once with the frozen weights.
"""
import argparse
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import chain, loaders, metrics
from .rerank import WEIGHTS, BM25, Dense, featurize, rank_ids, with_ce_scores

FEATS = Path(".cache/features")
GRID = {
    "dense": [0, 0.5, 1],
    "ce": [0, 0.5, 1, 2, 4],
    "scope_llm": [0, 0.5, 1, 2, 4],
    "bm25": [0, 0.25, 0.5, 1, 2],
    "scope": [0, 0.5, 1, 2, 4],
    "entity": [0, 0.5, 1, 2],
    "recency": [0, 0.25, 0.5, 1],
    "recency_cue": [0, 0.5, 1, 2, 4],
}
K = 10


def cmd_featurize(a):
    qs = {"longmemeval": loaders.load_longmemeval, "locomo": loaders.load_locomo,
          "domain": loaders.load_domain, "domain_dev": loaders.load_domain}[a.dataset](a.path, a.limit)
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


def tune(feats, rounds=3, use_ce=False, skip=("scope_llm",)):
    w = dict(WEIGHTS)
    best = objective(evaluate(feats, w))
    for _ in range(rounds):
        for name, grid in GRID.items():
            if (name == "ce" and not use_ce) or name in skip:
                continue
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


def cmd_cefeat(a):
    from sentence_transformers import CrossEncoder

    ce = CrossEncoder("BAAI/bge-reranker-base")
    path = {"longmemeval": a.lme_path, "locomo": a.locomo_path, "domain": a.domain_path,
            "domain_dev": "datasets/securitydomain_dev"}[a.dataset]
    qs = {"longmemeval": loaders.load_longmemeval, "locomo": loaders.load_locomo,
          "domain": loaders.load_domain, "domain_dev": loaders.load_domain}[a.dataset](path)
    by_id = {q.id: q for q in qs}
    feats = pickle.load(open(FEATS / f"{a.dataset}.pkl", "rb"))
    partial = FEATS / f"{a.dataset}_ce.partial.pkl"
    done = pickle.load(open(partial, "rb")) if partial.exists() else {}
    todo = [F for i, F in enumerate(feats) if i % a.stride == 0 and F["id"] not in done]
    for n, F in enumerate(todo):
        q = by_id[F["id"]]
        done[F["id"]] = with_ce_scores(
            {**F, "query": q.text}, {m.id: m.text for m in q.memories}, ce
        )
        if n % 25 == 0:
            print(f"{a.dataset} {n}/{len(todo)}", flush=True)
            pickle.dump(done, open(partial, "wb"))
    out = [done[F["id"]] for i, F in enumerate(feats) if i % a.stride == 0]
    pickle.dump(out, open(FEATS / f"{a.dataset}_ce.pkl", "wb"))


def attach_scopes(feats, path):
    """Add in_scope_llm from a labels file of {id, start, end} absolute-date windows."""
    labels = {r["id"]: r for r in map(json.loads, open(path))}
    day = lambda d: datetime.fromisoformat(d).replace(tzinfo=timezone.utc).timestamp()
    for F in feats:
        r = labels[F["id"]]
        F["has_scope_llm"] = r["start"] is not None
        if F["has_scope_llm"]:
            lo, hi = day(r["start"]), day(r["end"]) + 86400 - 1
            F["in_scope_llm"] = ((F["ts"] >= lo) & (F["ts"] <= hi)).astype(float)
        else:
            F["in_scope_llm"] = np.zeros(len(F["ts"]))


def scope_diagnostics(name, feats):
    from collections import Counter

    n, rx, llm, both = Counter(), Counter(), Counter(), Counter()
    hit, scoped = 0, 0
    for F in feats:
        n[F["qtype"]] += 1
        rx[F["qtype"]] += bool(F["has_scope"])
        llm[F["qtype"]] += bool(F["has_scope_llm"])
        both[F["qtype"]] += bool(F["has_scope"] and F["has_scope_llm"])
        if F["has_scope_llm"] and F["gold"]:
            scoped += 1
            gold_in = [F["in_scope_llm"][i] for i, mid in enumerate(F["ids"]) if mid in F["gold"]]
            hit += bool(gold_in) and max(gold_in) > 0
    lines = [f"\n### {name}: scope coverage (questions with a window)\n",
             "| type | n | regex | LLM | both |", "|---|---|---|---|---|"]
    for t in sorted(n):
        lines.append(f"| {t} | {n[t]} | {rx[t]} | {llm[t]} | {both[t]} |")
    lines.append(f"| ALL | {sum(n.values())} | {sum(rx.values())} | {sum(llm.values())} | "
                 f"{sum(both.values())} |")
    if scoped:
        lines.append(f"\nOf {scoped} questions with an LLM window and gold evidence in the pool, "
                     f"{hit} ({hit / scoped:.0%}) have at least one gold turn inside the window.")
    return "\n".join(lines)


def cmd_tune_llm(a):
    lme = pickle.load(open(FEATS / "longmemeval_ce.pkl", "rb"))
    loc = pickle.load(open(FEATS / "locomo_ce.pkl", "rb"))
    attach_scopes(lme, a.lme_labels)
    attach_scopes(loc, a.locomo_labels)
    fit, dev = lme[0::2], lme[1::2]
    w_rx, _ = tune(fit, use_ce=True)
    w_llm, fit_score = tune(fit, use_ce=True, skip=())
    out = ["# Stage 2: LLM time-scope parse vs regex (k=10, retrieval only)\n",
           "Labels come from a Haiku subagent that saw only each question and its date "
           "(no haystack, no answer). Same pool, split and protocol as `stage2_ce.md`.\n",
           f"Tuned with CE, regex scope only: `{w_rx}`\n",
           f"Tuned with CE + LLM scope: `{w_llm}` (fit objective {fit_score:.3f})\n"]
    cfgs = [("CE only", {**WEIGHTS, "dense": 0.0, "ce": 1.0}),
            ("tuned, CE + regex scope", w_rx), ("tuned, CE + LLM scope", w_llm)]
    if w_llm["scope_llm"]:
        cfgs.append(("CE + LLM scope, minus scope_llm", {**w_llm, "scope_llm": 0.0}))
    for name, feats in (("LongMemEval_s dev half", dev), ("LoCoMo (held out, every 4th q)", loc)):
        out.append(f"\n## {name}\n\n| config | recall@10 | MRR |\n|---|---|---|")
        for label, cw in cfgs:
            rows = evaluate(feats, cw)
            rr, mm = metrics.mean_ci([x[1] for x in rows]), metrics.mean_ci([x[2] for x in rows])
            out.append(f"| {label} | {rr[0]:.3f} [{rr[1]:.3f}, {rr[2]:.3f}] | "
                       f"{mm[0]:.3f} [{mm[1]:.3f}, {mm[2]:.3f}] |")
        sub = [F for F in feats if F["has_scope_llm"]]
        out.append(f"\nOnly the {len(sub)} questions with an LLM window:\n\n"
                   "| config | recall@10 | MRR |\n|---|---|---|")
        for label, cw in cfgs:
            rows = evaluate(sub, cw)
            rr, mm = metrics.mean_ci([x[1] for x in rows]), metrics.mean_ci([x[2] for x in rows])
            if rr[0] is not None:
                out.append(f"| {label} | {rr[0]:.3f} [{rr[1]:.3f}, {rr[2]:.3f}] | "
                           f"{mm[0]:.3f} [{mm[1]:.3f}, {mm[2]:.3f}] |")
    out.append(scope_diagnostics("LongMemEval_s (all 500)", lme))
    out.append(scope_diagnostics("LoCoMo (497)", loc))
    text = "\n".join(out) + "\n"
    print(text)
    Path("results/stage2_llmscope.md").write_text(text)


# Weights fit on even-indexed LongMemEval_s only (results/stage2_ce.md); never retuned on the domain set.
FROZEN = {"dense": 1.0, "ce": 2, "scope_llm": 0.0, "bm25": 2, "scope": 4, "entity": 0.0,
          "recency": 0.5, "recency_cue": 0.5}


DATASET_ARGS = {"longmemeval": (loaders.load_longmemeval, "data/longmemeval/longmemeval_s_cleaned.json"),
                "locomo": (loaders.load_locomo, "data/locomo/locomo10.json"),
                "domain": (loaders.load_domain, "datasets/securitydomain"),
                "domain_dev": (loaders.load_domain, "datasets/securitydomain_dev")}


def cmd_fullfeat(a):
    load, path = DATASET_ARGS[a.dataset]
    qs = load(path)
    dense, bm25, cache, out = Dense(), BM25(), {}, {}
    for i, q in enumerate(qs):
        if i % a.stride == 0:
            out[q.id] = chain.build_full(q, dense, bm25, cache)
        if i % 50 == 0:
            print(f"{a.dataset} {i}/{len(qs)}", flush=True)
    pickle.dump(out, open(FEATS / f"{a.dataset}_full.pkl", "wb"))


CHAIN_GRID = {"w_upd": [0, 0.5, 1, 2, 4], "w_upd_cue": [0, 0.5, 1, 2, 4], "margin": [1, 2, 4, 8],
              "hops": [0, 1, 2, 3], "df_max": [20, 40, 80], "seeds": [1, 2]}


def load_set(name):
    feats = pickle.load(open(FEATS / f"{name}_ce.pkl", "rb"))
    fulls = pickle.load(open(FEATS / f"{name}_full.pkl", "rb"))
    return [F for F in feats if F["id"] in fulls], fulls


def eval_chain(feats, fulls, p):
    rows = []
    for F in feats:
        ids = chain.rank(F, fulls[F["id"]], FROZEN, p)[:K]
        rows.append((F["qtype"], metrics.recall_at_k(ids, F["gold"], K), metrics.mrr(ids, F["gold"])))
    return rows


def cmd_tune_chain(a):
    grid = {k: ([0] if (a.gate_update and k == "w_upd") else v) for k, v in CHAIN_GRID.items()}
    dev, dev_full = load_set("domain_dev")
    p, best = dict(chain.DEFAULT), objective(eval_chain(dev, dev_full, chain.DEFAULT))
    for _ in range(3):
        for name, values in grid.items():
            for v in values:
                s = objective(eval_chain(dev, dev_full, {**p, name: v}))
                if s > best + 1e-6:
                    best, p = s, {**p, name: v}
    base = dict(chain.DEFAULT)
    out = ["# Entity-scoped recency and entity chaining (k=10, retrieval only, no training)\n",
           "Base scorer = the Stage 2 weights frozen from LongMemEval_s. The two new signals' parameters "
           f"are tuned ONLY on a second synthetic set (seed 11, {len(dev)} questions); evaluation sets "
           "were never used for tuning" + (" (update bonus restricted to recency-cue questions; this restriction was chosen after seeing the ungated no-harm result). " if a.gate_update else ". ") + "Domain sets share a generator, so this is in-distribution; "
           "LoCoMo and LongMemEval_s are no-harm checks.\n",
           f"Tuned parameters: `{p}` (dev objective {best:.3f}, base {objective(eval_chain(dev, dev_full, base)):.3f})\n"]
    cfgs = [("base (frozen Stage 2)", base),
            ("+ entity-scoped recency", {**base, **{k: p[k] for k in ("w_upd", "w_upd_cue", "margin")}}),
            ("+ entity chaining", {**base, **{k: p[k] for k in ("hops", "df_max", "seeds")}}),
            ("+ both", p)]
    for name, key, note in (("Domain test set (seed 7)", "domain", ""),
                            ("LoCoMo (every 4th question; no-harm check)", "locomo", ""),
                            ("LongMemEval_s (every 4th question; no-harm check)", "longmemeval", "")):
        feats, fulls = load_set(key)
        out.append(f"\n## {name}, n={len(feats)}\n\n| config | recall@10 | MRR |\n|---|---|---|")
        for label, cp in cfgs:
            rows = eval_chain(feats, fulls, cp)
            rr, mm = metrics.mean_ci([x[1] for x in rows]), metrics.mean_ci([x[2] for x in rows])
            out.append(f"| {label} | {rr[0]:.3f} [{rr[1]:.3f}, {rr[2]:.3f}] | {mm[0]:.3f} [{mm[1]:.3f}, {mm[2]:.3f}] |")
        for label, cp in (cfgs[0], cfgs[3]):
            out.append(f"\nPer type, {label}:\n\n{table(eval_chain(feats, fulls, cp))}")
    text = "\n".join(out) + "\n"
    print(text)
    Path("results/stage3_chain_gated.md" if a.gate_update else "results/stage3_chain.md").write_text(text)


def cmd_tune_domain(a):
    feats = pickle.load(open(FEATS / "domain_ce.pkl", "rb"))
    cfgs = [("dense only", dict(WEIGHTS)), ("dense+bm25 (w=1)", {**WEIGHTS, "bm25": 1.0}),
            ("CE only", {**WEIGHTS, "dense": 0.0, "ce": 1.0}),
            ("frozen LME-tuned, with CE", FROZEN),
            ("frozen minus scope", {**FROZEN, "scope": 0.0}),
            ("frozen minus recency+cue", {**FROZEN, "recency": 0.0, "recency_cue": 0.0}),
            ("frozen + entity 1 (exploratory)", {**FROZEN, "entity": 1.0}),
            ("frozen + entity 2 (exploratory)", {**FROZEN, "entity": 2.0})]
    out = ["# Stage 3: security-domain set (k=10, retrieval only, no training)\n",
           f"{len(feats)} questions over a {3069}-record synthetic history. Weights are frozen from "
           "LongMemEval_s; nothing is tuned on this set. Pool: fused top-30 plus dense and BM25 top-10.\n",
           "| config | recall@10 | MRR |\n|---|---|---|"]
    for label, cw in cfgs:
        rows = evaluate(feats, cw)
        rr, mm = metrics.mean_ci([x[1] for x in rows]), metrics.mean_ci([x[2] for x in rows])
        out.append(f"| {label} | {rr[0]:.3f} [{rr[1]:.3f}, {rr[2]:.3f}] | {mm[0]:.3f} [{mm[1]:.3f}, {mm[2]:.3f}] |")
    for label, cw in (cfgs[2], cfgs[3], cfgs[6]):
        out.append(f"\nPer type, {label}:\n\n{table(evaluate(feats, cw))}")
    text = "\n".join(out) + "\n"
    print(text)
    Path("results/stage3_rerank.md").write_text(text)


def cmd_tune_ce(a):
    lme = pickle.load(open(FEATS / "longmemeval_ce.pkl", "rb"))
    loc = pickle.load(open(FEATS / "locomo_ce.pkl", "rb"))
    fit, dev = lme[0::2], lme[1::2]
    w_plain, _ = tune(fit)
    w_ce, fit_score = tune(fit, use_ce=True)
    out = ["# Stage 2 spike with cross-encoder feature (k=10, retrieval only, no training)\n",
           "Pool per question: fused (dense+BM25) top-30 plus dense and BM25 top-10 "
           "(same pool for every row). Weights fit on even-indexed LongMemEval_s only; "
           "dev = odd-indexed; LoCoMo scored once with frozen weights.\n",
           f"Tuned without CE: `{w_plain}`\n",
           f"Tuned with CE: `{w_ce}` (fit objective {fit_score:.3f})\n"]
    cfgs = [("dense only", dict(WEIGHTS)), ("dense+bm25 (w=1)", {**WEIGHTS, "bm25": 1.0}),
            ("CE only", {**WEIGHTS, "dense": 0.0, "ce": 1.0}),
            ("tuned, no CE", w_plain), ("tuned, with CE", w_ce)]
    cfgs += [(f"with CE, minus {k}", {**w_ce, k: 0.0}) for k in ("ce", "bm25", "scope", "entity",
                                                              "recency", "recency_cue") if w_ce[k]]
    for name, feats in (("LongMemEval_s dev half", dev), ("LoCoMo (held out)", loc)):
        out.append(f"\n## {name}\n\n| config | recall@10 | MRR |\n|---|---|---|")
        for label, cw in cfgs:
            rows = evaluate(feats, cw)
            rr, mm = metrics.mean_ci([x[1] for x in rows]), metrics.mean_ci([x[2] for x in rows])
            out.append(f"| {label} | {rr[0]:.3f} [{rr[1]:.3f}, {rr[2]:.3f}] | "
                       f"{mm[0]:.3f} [{mm[1]:.3f}, {mm[2]:.3f}] |")
        out.append(f"\nPer type, tuned with CE:\n\n{table(evaluate(feats, w_ce))}")
    text = "\n".join(out) + "\n"
    print(text)
    Path("results/stage2_ce.md").write_text(text)


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
    f.add_argument("--dataset", choices=["longmemeval", "locomo", "domain", "domain_dev"], required=True)
    f.add_argument("--path", required=True)
    f.add_argument("--limit", type=int, default=0)
    f.set_defaults(fn=cmd_featurize)
    t = sub.add_parser("tune")
    t.set_defaults(fn=cmd_tune)
    c = sub.add_parser("cefeat")
    c.add_argument("--dataset", choices=["longmemeval", "locomo", "domain", "domain_dev"], required=True)
    c.add_argument("--domain-path", default="datasets/securitydomain")
    c.add_argument("--lme-path", default="data/longmemeval/longmemeval_s_cleaned.json")
    c.add_argument("--locomo-path", default="data/locomo/locomo10.json")
    c.add_argument("--stride", type=int, default=1, help="score every Nth question")
    c.set_defaults(fn=cmd_cefeat)
    t3 = sub.add_parser("tune-llm")
    t3.add_argument("--lme-labels", default="data/scopes/lme_out.jsonl")
    t3.add_argument("--locomo-labels", default="data/scopes/locomo_out.jsonl")
    t3.set_defaults(fn=cmd_tune_llm)
    sub.add_parser("tune-domain").set_defaults(fn=cmd_tune_domain)
    ff = sub.add_parser("fullfeat")
    ff.add_argument("--dataset", choices=sorted(DATASET_ARGS), required=True)
    ff.add_argument("--stride", type=int, default=1)
    ff.set_defaults(fn=cmd_fullfeat)
    tc = sub.add_parser("tune-chain")
    tc.add_argument("--gate-update", action="store_true",
                    help="allow the update bonus only on questions with a recency cue")
    tc.set_defaults(fn=cmd_tune_chain)
    t2 = sub.add_parser("tune-ce")
    t2.set_defaults(fn=cmd_tune_ce)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
