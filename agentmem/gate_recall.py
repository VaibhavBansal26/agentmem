"""CI gate: retrieval quality of the full pipeline on a fixed 30-question subset.

    python -m agentmem.gate_recall            # compare with ci/recall_baseline.json, exit 1 on regression
    python -m agentmem.gate_recall --update   # recompute and overwrite the baseline (review the diff!)

No LLM and no API key: it measures what the memory server would hand an agent. Fails if recall@10 or
MRR drops more than TOLERANCE below the committed baseline, or recall@10 falls under FLOOR.
"""
import json
import sys
from collections import Counter
from pathlib import Path

from . import metrics
from .domain import ASKED_AT
from .loaders import load_domain
from .service import MemoryService

SUBSET = Path("ci/recall_subset.json")
BASELINE = Path("ci/recall_baseline.json")
TOLERANCE, FLOOR, K = 0.03, 0.80, 10
PER_TYPE = {"single-hop": 6, "multi-hop": 8, "temporal": 8, "knowledge-update": 8}


def make_subset(questions):
    """First N questions of each type in id order: fixed, stratified, no randomness."""
    taken, ids = Counter(), []
    for q in sorted(questions, key=lambda q: q.id):
        if taken[q.qtype] < PER_TYPE.get(q.qtype, 0):
            taken[q.qtype] += 1
            ids.append(q.id)
    return ids


def evaluate():
    questions = load_domain("datasets/securitydomain")
    if not SUBSET.exists():
        SUBSET.parent.mkdir(exist_ok=True)
        SUBSET.write_text(json.dumps(make_subset(questions), indent=1) + "\n")
    wanted = set(json.loads(SUBSET.read_text()))
    chosen = [q for q in questions if q.id in wanted]
    svc = MemoryService(now=ASKED_AT, rerank=True, use_ce=True)
    svc.bulk_load(chosen[0].memories)
    rows = []
    for q in chosen:
        ids = [x["id"] for x in svc.recall(q.text, k=K)]
        rows.append((q.qtype, metrics.recall_at_k(ids, q.gold_ids, K), metrics.mrr(ids, q.gold_ids)))
    mean = lambda xs: sum(xs) / len(xs)
    return {"n": len(rows), "recall@10": round(mean([r[1] for r in rows]), 4), "mrr": round(mean([r[2] for r in rows]), 4),
            "by_type": {t: round(mean([r[1] for r in rows if r[0] == t]), 4) for t in sorted({r[0] for r in rows})}}


def main():
    result = evaluate()
    print(json.dumps(result, indent=1))
    if "--update" in sys.argv:
        BASELINE.write_text(json.dumps(result, indent=1) + "\n")
        print("baseline updated")
        return 0
    base = json.loads(BASELINE.read_text())
    problems = []
    for key in ("recall@10", "mrr"):
        if result[key] < base[key] - TOLERANCE:
            problems.append(f"{key} {result[key]} is more than {TOLERANCE} below baseline {base[key]}")
    if result["recall@10"] < FLOOR:
        problems.append(f"recall@10 {result['recall@10']} is under the floor {FLOOR}")
    for p in problems:
        print("GATE FAILED:", p)
    print("GATE PASSED" if not problems else "")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
