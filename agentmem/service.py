"""Memory store + recall pipeline behind the MCP server (no MCP imports here, so it is easy to test).

Variant A (rerank=False) is the dense top-k baseline. Variant B (rerank=True) is the Stage 2/3
pipeline: dense+BM25 candidates, cross-encoder score, cue-gated entity-scoped recency, entity chaining.
"""
import functools
import json
import sqlite3
import threading
import time
import uuid
from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np

from . import chain
from .rerank import featurize, with_ce_scores
from .retrievers import BM25, Dense
from .schema import Memory

# Fit on LongMemEval_s (results/stage2_ce.md); chain parameters tuned on the seed-11 synthetic set
# with the update bonus gated on recency cues (results/stage3_chain_gated.md).
WEIGHTS = {"dense": 1.0, "ce": 2, "scope_llm": 0.0, "bm25": 2, "scope": 4, "entity": 0.0,
           "recency": 0.5, "recency_cue": 0.5}
CHAIN = {"w_upd": 0.0, "w_upd_cue": 4, "margin": 4, "hops": 3, "df_max": 40, "seeds": 1}


def locked(fn):
    """Serialize a MemoryService method: the MCP server calls tools from worker threads."""

    @functools.wraps(fn)
    def wrapper(self, *a, **kw):
        with self._lock:
            return fn(self, *a, **kw)

    return wrapper


def iso(ts):
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_day(s, end=False):
    d = datetime.fromisoformat(s[:10]).replace(tzinfo=timezone.utc).timestamp()
    return d + 86399 if end else d


class MemoryService:
    def __init__(self, db=":memory:", rerank=True, use_ce=True, now=None):
        self.db = sqlite3.connect(db, check_same_thread=False)
        self.db.execute("""CREATE TABLE IF NOT EXISTS memories (
            id TEXT PRIMARY KEY, text TEXT, ts REAL, source TEXT, entities TEXT, importance REAL)""")
        self.rerank_on, self.use_ce, self.now = rerank, use_ce, now
        self._lock = threading.RLock()  # tool calls arrive on worker threads; models are shared
        self._dense = self._bm25 = self._ce = None
        self._load()

    # --- storage ---------------------------------------------------------------------------
    def _load(self):
        self.memories = [
            Memory(i, t, ts, s, tuple(json.loads(e)), imp)
            for i, t, ts, s, e, imp in self.db.execute(
                "SELECT id, text, ts, source, entities, importance FROM memories ORDER BY ts")
        ]

    def clock(self):
        return self.now if self.now is not None else time.time()

    @locked
    def store(self, text, source="agent_note", entities=(), importance=0.0, timestamp=None, mid=None):
        m = Memory(mid or f"m{uuid.uuid4().hex[:10]}", text,
                   timestamp if timestamp is not None else self.clock(), source, tuple(entities), importance)
        self.db.execute("INSERT OR REPLACE INTO memories VALUES (?,?,?,?,?,?)",
                        (m.id, m.text, m.timestamp, m.source, json.dumps(list(m.entities)), m.importance))
        self.db.commit()
        self.memories = sorted([x for x in self.memories if x.id != m.id] + [m], key=lambda x: x.timestamp)
        return m.id

    @locked
    def bulk_load(self, memories):
        self.db.executemany("INSERT OR REPLACE INTO memories VALUES (?,?,?,?,?,?)", [
            (m.id, m.text, m.timestamp, m.source, json.dumps(list(m.entities)), m.importance)
            for m in memories])
        self.db.commit()
        self._load()

    # --- retrieval -------------------------------------------------------------------------
    def _models(self, need_ce):
        if self._dense is None:
            self._dense, self._bm25 = Dense(), BM25()
        if need_ce and self.use_ce and self._ce is None:
            from sentence_transformers import CrossEncoder

            self._ce = CrossEncoder("BAAI/bge-reranker-base")

    def _view(self, query, mems, ts_now):
        return SimpleNamespace(id="live", text=query, qtype="live", gold_ids=(), memories=mems,
                               asked_at=ts_now)

    def _out(self, mems, ranked_ids, k):
        by_id = {m.id: m for m in mems}
        return [{"rank": r + 1, "id": i, "text": by_id[i].text, "timestamp": iso(by_id[i].timestamp),
                 "source": by_id[i].source} for r, i in enumerate(ranked_ids[:k])]

    def _rank(self, query, mems, k, rerank):
        """Rank `mems` for `query` as of the service clock; returns a list of ids."""
        ts_now = self.clock()
        mems = [m for m in mems if m.timestamp <= ts_now]
        if not mems:
            return mems, []
        self._models(rerank)
        q = self._view(query, mems, ts_now)
        if not rerank:
            scores = self._dense.scores(query, mems)
            return mems, [mems[i].id for i in np.argsort(-scores, kind="stable")[:k]]
        F = featurize(q, self._dense, self._bm25)
        if self.use_ce:
            F = with_ce_scores(F, {m.id: m.text for m in mems}, self._ce)
        full = chain.build_full(q, self._dense, self._bm25, {})
        return mems, chain.rank(F, full, WEIGHTS, CHAIN)[:k]

    @locked
    def recall(self, query, k=8, start_date=None, end_date=None, rerank=None):
        mems = self.memories
        if start_date or end_date:
            lo = parse_day(start_date) if start_date else -1e18
            hi = parse_day(end_date, end=True) if end_date else 1e18
            mems = [m for m in mems if lo <= m.timestamp <= hi]
        mems, ids = self._rank(query, mems, k, self.rerank_on if rerank is None else rerank)
        return self._out(mems, ids, k)

    @locked
    def rerank_candidates(self, query, candidates, k=None):
        """Rerank caller-supplied candidates ({text, optional timestamp ISO, optional id})."""
        mems = []
        for n, c in enumerate(candidates):
            ts = parse_day(c["timestamp"]) if c.get("timestamp") else self.clock()
            mems.append(Memory(str(c.get("id", f"c{n}")), c["text"], ts, c.get("source", "candidate")))
        mems, ids = self._rank(query, mems, k or len(mems), True)
        return self._out(mems, ids, k or len(mems))
