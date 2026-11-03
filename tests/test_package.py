import asyncio
import importlib.util
import json
import os
import tempfile
import warnings
import zlib

import numpy as np

from agentmem import Config, Memory
from agentmem.adapters import anthropic_tools, openai_agents_tools


def bow(texts):
    """Deterministic bag-of-words embedder (no model needed)."""
    out = np.zeros((len(texts), 128))
    for i, t in enumerate(texts):
        for w in t.lower().split():
            out[i, zlib.crc32(w.encode()) % 128] += 1
    return out


def overlap(pairs):
    return [float(len(set(q.lower().split()) & set(t.lower().split()))) for q, t in pairs]


def make(path=":memory:", **kw):
    return Memory(path, embedder=bow, scorer=overlap, **kw)


def test_update_handling_and_ranking_with_plugins():
    mem = make()
    mem.add("Host web-12 role set to staging web server", timestamp="2026-10-02")
    mem.add("Host web-12 role changed from staging web server to production web server", timestamp="2026-10-20")
    mem.add("Antivirus scan completed on web-12 with no findings", timestamp="2026-10-25")
    hits = mem.recall("What is the current role of web-12?", k=3)
    assert [h.rank for h in hits] == [1, 2, 3]
    assert "production" in hits[0].text  # the latest version outranks the older one
    assert len(mem) == 3


def test_scorer_decides_between_equally_relevant_candidates():
    """The scorer is one signal in a weighted blend (not an override): it must decide between candidates the
    lexical and dense signals cannot separate, in whichever direction it points."""

    def top(prefer):
        m = Memory(embedder=bow, scorer=lambda pairs: [5.0 if prefer in t else 0.0 for _, t in pairs])
        for i in range(20):
            m.add(f"routine filler {i} on host-{i} finished", timestamp="2026-10-01")
        m.add("alert about host-3 handled by team alpha", timestamp="2026-10-01")
        m.add("alert about host-3 handled by team beta", timestamp="2026-10-01")
        return m.recall("alert about host-3 handled by team", k=1)[0].text

    assert top("beta").endswith("beta") and top("alpha").endswith("alpha")


def test_embedder_signatures_are_not_misread():
    """An optional second parameter (e.g. dim) must not be mistaken for the is_query flag."""
    dims, flags = [], []

    def with_dim(texts, dim=64):  # an optional extra that is NOT is_query: must always keep its default
        dims.append(dim)
        return bow(texts)[:, :dim]

    def with_flag(texts, is_query=False):
        flags.append(is_query)
        return bow(texts)

    for fn in (with_dim, with_flag):
        mem = Memory(embedder=fn, scorer=overlap)
        mem.add("host db-1 patched", timestamp="2026-10-01")
        assert mem.recall("db-1 patched", k=1)[0].text == "host db-1 patched"
    assert dims and set(dims) == {64}  # never overwritten by a boolean
    assert set(flags) == {True, False}  # passages embedded with False, the query with True


def test_persistence_delete_and_window():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "m.db")
        with make(path) as mem:
            a = mem.add("firewall rule opened for 198.51.100.7", timestamp="2026-10-03")
            mem.add("firewall rule closed for 198.51.100.7", timestamp="2026-10-20")
        with make(path) as mem:  # reopened: data persisted
            assert len(mem) == 2
            assert [h.text for h in mem.recall("firewall 198.51.100.7", k=5, start="2026-10-15")] == [
                "firewall rule closed for 198.51.100.7"]
            assert mem.get(a).text.startswith("firewall rule opened")
            assert mem.delete(a) and not mem.delete(a) and len(mem) == 1
        assert sorted(os.listdir(d)) == ["m.db"]  # custom embedder: no cache file next to the database


def test_clock_hides_future_memories_and_no_cache_dir_is_created():
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as d:
        os.chdir(d)
        try:
            mem = make(now="2026-10-10")
            mem.add("early note about db-1", timestamp="2026-10-05")
            mem.add("late note about db-1", timestamp="2026-10-20")
            assert [h.text for h in mem.recall("note about db-1")] == ["early note about db-1"]
            assert os.listdir(d) == []  # nothing written into the caller's working directory
        finally:
            os.chdir(cwd)


def test_add_many_and_timestamp_types():
    from datetime import date, datetime, timezone

    mem = make()
    ids = mem.add_many([{"text": "a one", "id": "x1", "timestamp": date(2026, 10, 1)},
                        {"text": "a two", "timestamp": datetime(2026, 10, 2, tzinfo=timezone.utc)},
                        {"text": "a three", "timestamp": 1790900000.0}])
    assert ids[0] == "x1" and len(set(ids)) == 3 and len(mem) == 3


def test_missing_extra_warns_instead_of_failing():
    real = importlib.util.find_spec
    importlib.util.find_spec = lambda name, *a, **k: None if name == "sentence_transformers" else real(name, *a, **k)
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            Memory()
        assert any("sentence-transformers is not installed" in str(x.message) for x in w)
    finally:
        importlib.util.find_spec = real


def test_matches_the_evaluated_pipeline_on_the_domain_set():
    """The public class must rank exactly like the pipeline whose results are reported."""
    from agentmem.domain import ASKED_AT
    from agentmem.loaders import load_domain
    from agentmem.service import MemoryService

    qs = load_domain("datasets/securitydomain")
    svc = MemoryService(now=ASKED_AT, rerank=True, use_ce=True)
    svc.bulk_load(qs[0].memories)
    mem = Memory(now=ASKED_AT)
    mem.add_many([{"id": m.id, "text": m.text, "timestamp": m.timestamp, "source": m.source,
                   "entities": m.entities, "importance": m.importance} for m in qs[0].memories])
    sample = [q for q in qs if q.qtype != "abstention"][::12][:7]
    assert sample
    for q in sample:
        assert [h.id for h in mem.recall(q.text, k=10)] == [x["id"] for x in svc.recall(q.text, k=10)], q.id


def test_anthropic_adapter_round_trip():
    mem = make()
    tools, handle = anthropic_tools(mem)
    assert [t["name"] for t in tools] == ["memory_recall", "memory_store"]
    assert all(t["input_schema"]["type"] == "object" for t in tools)
    stored = json.loads(handle("memory_store", {"text": "ticket T-9 closed: false positive", "entities": ["T-9"]}))
    hits = json.loads(handle("memory_recall", {"query": "outcome of ticket T-9", "k": 2}))
    assert hits[0]["id"] == stored["id"] and hits[0]["rank"] == 1


def test_openai_agents_adapter_invokes_tools():
    from agents.tool_context import ToolContext

    mem = make()
    by_name = {t.name: t for t in openai_agents_tools(mem)}
    assert set(by_name) == {"memory_recall", "memory_store"}

    def call(name, args):
        raw = json.dumps(args)
        ctx = ToolContext(context=None, tool_name=name, tool_call_id="c1", tool_arguments=raw)
        return asyncio.run(by_name[name].on_invoke_tool(ctx, raw))

    stored = json.loads(call("memory_store", {"text": "host db-9 patched"}))
    hits = json.loads(call("memory_recall", {"query": "db-9 patched"}))
    assert hits[0]["id"] == stored["id"] and len(mem) == 1


if __name__ == "__main__":
    test_update_handling_and_ranking_with_plugins()
    test_scorer_decides_between_equally_relevant_candidates()
    test_embedder_signatures_are_not_misread()
    test_persistence_delete_and_window()
    test_clock_hides_future_memories_and_no_cache_dir_is_created()
    test_add_many_and_timestamp_types()
    test_missing_extra_warns_instead_of_failing()
    test_anthropic_adapter_round_trip()
    test_openai_agents_adapter_invokes_tools()
    test_matches_the_evaluated_pipeline_on_the_domain_set()
    print("package ok")
