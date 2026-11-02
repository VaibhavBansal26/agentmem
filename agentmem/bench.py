import argparse
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from . import loaders, metrics, retrievers
from .llm import answer, build_prompt


def load(dataset, path, limit):
    if dataset == "toy":
        return loaders.load_toy(limit)
    if not path:
        raise SystemExit(f"--path is required for {dataset}")
    return {"longmemeval": loaders.load_longmemeval, "locomo": loaders.load_locomo}[dataset](
        path, limit
    )


def run(questions, retriever, k, use_answerer):
    rows = []
    for q in questions:
        ranked = retriever.retrieve(q.text, q.memories, k, now=q.asked_at)
        ids = [m.id for m in ranked]
        row = {
            "qtype": q.qtype,
            "recall": metrics.recall_at_k(ids, q.gold_ids, k),
            "mrr": metrics.mrr(ids, q.gold_ids),
            "f1": None,
        }
        if use_answerer:
            now = datetime.fromtimestamp(q.asked_at, timezone.utc).strftime("%Y-%m-%d")
            pred = answer(build_prompt(q.text, ranked, now))
            row["f1"] = metrics.token_f1(pred, q.answer)
        rows.append(row)
    return rows


def fmt(vals):
    mean, lo, hi, n = metrics.mean_ci(vals)
    return "n/a" if mean is None else f"{mean:.3f} [{lo:.3f}, {hi:.3f}]"


def table(rows, k, use_answerer):
    groups = defaultdict(list)
    for r in rows:
        groups[r["qtype"]].append(r)
        groups["ALL"].append(r)
    head = f"| type | n | recall@{k} | MRR |" + (" F1 |" if use_answerer else "")
    lines = [head, "|" + "---|" * (head.count("|") - 1)]
    for t in sorted(groups, key=lambda t: (t == "ALL", t)):
        g = groups[t]
        cells = [t, str(len(g)), fmt([r["recall"] for r in g]), fmt([r["mrr"] for r in g])]
        if use_answerer:
            cells.append(fmt([r["f1"] for r in g]))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["toy", "longmemeval", "locomo"], default="toy")
    ap.add_argument("--path")
    ap.add_argument("--retriever", default="bm25")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--answerer", action="store_true", help="also run the (paid, cached) answerer")
    a = ap.parse_args()

    retriever = retrievers.get(a.retriever)
    backend = getattr(retriever, "backend", "n/a")
    rows = run(load(a.dataset, a.path, a.limit), retriever, a.k, a.answerer)
    out = f"## {a.dataset} / {a.retriever} (backend: {backend})\n\n{table(rows, a.k, a.answerer)}\n"
    print(out)
    Path("results").mkdir(exist_ok=True)
    Path(f"results/{a.dataset}_{a.retriever}.md").write_text(out)


if __name__ == "__main__":
    main()
