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
from .index import IndexedBM25, IndexedDense, StoreIndex
from .retrievers import BM25, Dense
from .schema import Memory

# Fit on LongMemEval_s (results/stage2_ce.md); chain parameters tuned on the seed-11 synthetic set
# with the update bonus gated on recency cues (results/stage3_chain_gated.md).
WEIGHTS = {"dense": 1.0, "ce": 2, "scope_llm": 0.0, "bm25": 2, "scope": 4, "entity": 0.0,
           "recency": 0.5, "recency_cue": 0.5}
CHAIN = {"w_upd": 0.0, "w_upd_cue": 4, "margin": 4, "hops": 3, "df_max": 40, "seeds": 1}
# Without a cross-encoder the cross-encoder weight is meaningless: use the weights tuned without it
# (results/stage2_ce.md, "Tuned without CE").
LITE_WEIGHTS = {"dense": 1.0, "ce": 0.0, "scope_llm": 0.0, "bm25": 1, "scope": 2, "entity": 0.0,
                "recency": 0.0, "recency_cue": 0.0}


class FnScorer:
    """Adapts a callable scorer(pairs: list[(query, text)]) -> list[float] to the .predict() interface."""

    def __init__(self, fn):
        self.fn = fn

    def predict(self, pairs, batch_size=32, show_progress_bar=False):
        return np.asarray(self.fn([tuple(p) for p in pairs]), dtype=float)


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


class _Texts:
    """id -> text lookup over the indexed store without building a dict of every memory per query."""

    def __init__(self, mems, row):
        self.mems, self.row = mems, row

    def __getitem__(self, mid):
        return self.mems[self.row[mid]].text


class MemoryService:
    def __init__(self, db=":memory:", rerank=True, use_ce=True, now=None, *, dense=None, bm25=None,
                 scorer=None, weights=None, chain_params=None, cache_path=None, dense_kwargs=None,
                 ce_model="BAAI/bge-reranker-base", use_index=True):
        self.db = sqlite3.connect(db, check_same_thread=False)
        self.db.execute("""CREATE TABLE IF NOT EXISTS memories (
            id TEXT PRIMARY KEY, text TEXT, ts REAL, source TEXT, entities TEXT, importance REAL)""")
        self.rerank_on, self.use_ce, self.now = rerank, use_ce, now
        self._lock = threading.RLock()  # tool calls arrive on worker threads; models are shared
        self._dense, self._bm25 = dense, bm25
        self.use_index, self.index = use_index, StoreIndex()
        self._ce = FnScorer(scorer) if scorer is not None else None
        self._dense_kwargs = dense_kwargs if dense_kwargs is not None else {"cache_path": cache_path}
        self._ce_model = ce_model
        self.weights = dict(weights) if weights else dict(WEIGHTS if (use_ce or scorer is not None) else LITE_WEIGHTS)
        self.chain_params = dict(chain_params) if chain_params else dict(CHAIN)
        self._load()

    # --- storage ---------------------------------------------------------------------------
    def _load(self):
        rows = self.db.execute("SELECT id, text, ts, source, entities, importance FROM memories ORDER BY ts")
        self._reset([Memory(i, t, ts, s, tuple(json.loads(e)), imp) for i, t, ts, s, e, imp in rows])

    def _reset(self, memories):
        """Rebuild the derived index from a time-ordered list (O(N); done on open, bulk load, delete, backdated add)."""
        self.index.rebuild(memories)
        self.memories = self.index.memories

    def clock(self):
        return self.now if self.now is not None else time.time()

    @locked
    def store(self, text, source="agent_note", entities=(), importance=0.0, timestamp=None, mid=None):
        m = Memory(mid or f"m{uuid.uuid4().hex[:10]}", text,
                   timestamp if timestamp is not None else self.clock(), source, tuple(entities), importance)
        self.db.execute("INSERT OR REPLACE INTO memories VALUES (?,?,?,?,?,?)",
                        (m.id, m.text, m.timestamp, m.source, json.dumps(list(m.entities)), m.importance))
        self.db.commit()
        if m.id not in self.index.row and m.timestamp >= self.index.max_ts:
            self.index.append(m)  # in time order: update the index in place
        else:  # backdated or replacing an existing id: rebuild
            self._reset(sorted([x for x in self.index.memories if x.id != m.id] + [m], key=lambda x: x.timestamp))
        return m.id

    @locked
    def bulk_load(self, memories):
        self.db.executemany("INSERT OR REPLACE INTO memories VALUES (?,?,?,?,?,?)", [
            (m.id, m.text, m.timestamp, m.source, json.dumps(list(m.entities)), m.importance)
            for m in memories])
        self.db.commit()
        self._load()

    @locked
    def delete(self, mid):
        self.db.execute("DELETE FROM memories WHERE id = ?", (mid,))
        self.db.commit()
        if mid not in self.index.row:
            return False
        self._reset([m for m in self.index.memories if m.id != mid])
        return True

    @locked
    def get(self, mid):
        r = self.index.row.get(mid)
        return self.index.memories[r] if r is not None else None

    # --- retrieval -------------------------------------------------------------------------
    def _models(self, need_ce):
        if self._dense is None:
            self._dense = Dense(**self._dense_kwargs)
        if self._bm25 is None:
            self._bm25 = BM25()
        if self.use_index and not isinstance(self._dense, IndexedDense):
            self._dense, self._bm25 = IndexedDense(self.index, self._dense), IndexedBM25(self.index, self._bm25)
        if need_ce and self.use_ce and self._ce is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as e:
                raise ImportError("The cross-encoder needs sentence-transformers: pip install 'memorank[dense]', "
                                  "pass your own scorer=..., or construct with use_ce=False.") from e
            self._ce = CrossEncoder(self._ce_model)

    def _view(self, query, mems, ts_now, fast=False):
        view = SimpleNamespace(id="live", text=query, qtype="live", gold_ids=(), memories=mems, asked_at=ts_now)
        if fast:  # the whole indexed store is visible: hand over the precomputed structures
            ix = self.index
            view.all_visible, view.ids, view.tokens = True, ix.ids, ix.id_tokens
            view.df, view.row, view.id_post = ix.id_df, ix.row, ix.id_post
        return view

    def _out(self, mems, ranked_ids, k):
        if mems is self.index.memories:
            by_id = {i: mems[self.index.row[i]] for i in ranked_ids[:k]}
        else:
            by_id = {m.id: m for m in mems}
        return [{"rank": r + 1, "id": i, "text": by_id[i].text, "timestamp": iso(by_id[i].timestamp),
                 "source": by_id[i].source} for r, i in enumerate(ranked_ids[:k])]

    def _rank(self, query, mems, k, rerank):
        """Rank `mems` for `query` as of the service clock; returns a list of ids."""
        ts_now = self.clock()
        fast = self.use_index and mems is self.index.memories and self.index.max_ts <= ts_now
        if not fast:
            mems = [m for m in mems if m.timestamp <= ts_now]
        if not mems:
            return mems, []
        self._models(rerank)
        q = self._view(query, mems, ts_now, fast)
        if not rerank:
            scores = self._dense.scores(query, mems)
            return mems, [mems[i].id for i in np.argsort(-scores, kind="stable")[:k]]
        F = featurize(q, self._dense, self._bm25)
        if self._ce is not None or self.use_ce:
            texts = _Texts(mems, self.index.row) if fast else {m.id: m.text for m in mems}
            F = with_ce_scores(F, texts, self._ce)
        full = chain.build_full(q, self._dense, self._bm25, {})
        return mems, chain.rank(F, full, self.weights, self.chain_params)[:k]

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
