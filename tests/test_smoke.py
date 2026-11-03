from agentmem import loaders, metrics, retrievers
from agentmem.bench import run


def test_metrics():
    assert metrics.recall_at_k(["a", "b", "c"], ["b", "z"], 2) == 0.5
    assert metrics.mrr(["a", "b"], ["b"]) == 0.5
    assert metrics.recall_at_k(["a"], [], 5) is None
    assert metrics.token_f1("203.0.113.7", "203.0.113.7") == 1.0
    mean, lo, hi, n = metrics.mean_ci([1.0, 0.0, 1.0, 1.0])
    assert lo <= mean <= hi and n == 4


def test_baselines_run_on_toy():
    qs = loaders.load_toy()
    for name in ("recency", "bm25", "dense"):
        rows = run(qs, retrievers.get(name), k=5, use_answerer=False)
        assert len(rows) == len(qs)
        assert all(0.0 <= r["recall"] <= 1.0 for r in rows)


def test_bm25_beats_recency_on_toy():
    qs = loaders.load_toy()

    def avg(name):
        rows = run(qs, retrievers.get(name), k=3, use_answerer=False)
        return sum(r["recall"] for r in rows) / len(rows)

    assert avg("bm25") > avg("recency")


def test_longmemeval_no_gold_after_question_date():
    import os

    path = "data/longmemeval/longmemeval_s_cleaned.json"
    if not os.path.exists(path):
        return
    for q in loaders.load_longmemeval(path):
        ts = {m.id: m.timestamp for m in q.memories}
        assert all(ts[g] <= q.asked_at for g in q.gold_ids), q.id


def test_domain_generator_is_consistent():
    from agentmem import domain

    recs, qs = domain.build()
    assert (recs, qs) == domain.build()  # seeded: reproducible
    ids = {r["id"]: r for r in recs}
    assert len(ids) == len(recs) and all(r["ts"] <= domain.ASKED_AT for r in recs)
    for q in qs:
        assert all(g in ids for g in q["gold"]), q["id"]
        assert (q["qtype"] == "abstention") == (not q["gold"]), q["id"]


if __name__ == "__main__":
    test_domain_generator_is_consistent()
    test_longmemeval_no_gold_after_question_date()
    test_metrics()
    test_baselines_run_on_toy()
    test_bm25_beats_recency_on_toy()
    print("ok")
