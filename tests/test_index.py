"""The incremental index must be a pure speed-up: same scores, same rankings as the original code path."""
import random
import zlib

import numpy as np

from memorank.domain import ASKED_AT
from memorank.index import StoreIndex
from memorank.loaders import load_domain
from memorank.retrievers import BM25
from memorank.schema import Memory
from memorank.service import MemoryService


def bow(texts):
    out = np.zeros((len(texts), 96), dtype=np.float32)
    for i, t in enumerate(texts):
        for w in t.lower().split():
            out[i, zlib.crc32(w.encode()) % 96] += 1
    return out


def overlap(pairs):
    return [float(len(set(q.lower().split()) & set(t.lower().split()))) for q, t in pairs]


def test_bm25_scores_match_the_original_implementation():
    rnd = random.Random(1)
    words = ["alpha", "beta", "gamma", "host-1", "host-2", "login", "failed", "scan", "role", "ticket", "T-4001"]
    mems = [Memory(f"m{i}", " ".join(rnd.choices(words, k=rnd.randint(1, 12))), float(i)) for i in range(300)]
    appended, rebuilt = StoreIndex(), StoreIndex()
    for m in mems:
        appended.append(m)
    rebuilt.rebuild(mems)
    for q in ("failed login host-1", "role of host-2", "T-4001 ticket", "nothing matches zzz", "alpha alpha beta"):
        ref = np.asarray(BM25().scores(q, mems))
        assert np.allclose(appended.bm25(q), ref, atol=1e-9), q
        assert np.allclose(rebuilt.bm25(q), ref, atol=1e-9), q


def ranked(svc, qs, **kw):
    return [[x["id"] for x in svc.recall(q.text, k=10, **kw)] for q in qs]


def test_rankings_identical_with_and_without_the_index_on_the_domain_set():
    qs = [q for q in load_domain("datasets/securitydomain") if q.qtype != "abstention"]
    mems = qs[0].memories
    for kw, label in (({"use_ce": False}, "dense+BM25+signals"), ({"scorer": overlap, "use_ce": False}, "with a scorer")):
        fast = MemoryService(now=ASKED_AT, use_index=True, **kw)
        slow = MemoryService(now=ASKED_AT, use_index=False, **kw)
        fast.bulk_load(mems)
        slow.bulk_load(mems)
        a, b = ranked(fast, qs), ranked(slow, qs)
        assert a == b, (label, [i for i, (x, y) in enumerate(zip(a, b)) if x != y][:5])
    # the dense-only (reranker off) path too
    assert ranked(fast, qs, rerank=False) == ranked(slow, qs, rerank=False)


def test_rankings_identical_with_the_real_cross_encoder_on_a_sample():
    qs = [q for q in load_domain("datasets/securitydomain") if q.qtype != "abstention"][::9][:8]
    fast, slow = (MemoryService(now=ASKED_AT, use_index=u) for u in (True, False))
    for svc in (fast, slow):
        svc.bulk_load(qs[0].memories)
    assert ranked(fast, qs) == ranked(slow, qs)


def test_incremental_updates_match_a_fresh_build():
    base = [Memory(f"m{i}", f"host-{i % 7} event {i} login failed", float(i)) for i in range(60)]
    live = MemoryService(now=1e9, use_ce=False, scorer=overlap)
    live.bulk_load(base[:30])
    live.recall("host-3 login")  # forces the index and embedding matrix to exist before further changes
    for m in base[30:]:
        live.store(m.text, "agent_note", (), 0.0, m.timestamp, m.id)  # in time order: in-place appends
    live.store("host-3 role changed to production", "agent_note", (), 0.0, 5.0, "backdated")  # out of order: rebuild
    assert live.delete("m10") and not live.delete("m10") and live.get("m10") is None
    fresh = MemoryService(now=1e9, use_ce=False, scorer=overlap)
    fresh.bulk_load([m for m in live.memories])
    assert len(live.memories) == 60 and live.get("backdated").text.startswith("host-3 role")
    for q in ("host-3 login failed", "current role of host-3", "event 45"):
        assert [x["id"] for x in live.recall(q, k=8)] == [x["id"] for x in fresh.recall(q, k=8)], q


if __name__ == "__main__":
    test_bm25_scores_match_the_original_implementation()
    test_incremental_updates_match_a_fresh_build()
    test_rankings_identical_with_and_without_the_index_on_the_domain_set()
    test_rankings_identical_with_the_real_cross_encoder_on_a_sample()
    print("index ok")
