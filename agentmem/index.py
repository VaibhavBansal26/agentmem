"""Incremental, query-cheap view of a time-ordered memory list.

Everything here is derived data that does not depend on the query: BM25 term statistics and postings, identifier
tokens with document frequencies and postings, and a dense embedding matrix. It is updated in place when a memory
is appended in time order and rebuilt when one is inserted out of order or deleted. Scoring uses the same formulas
as `retrievers.BM25` and `retrievers.Dense`, just without re-tokenizing the whole store on every query.
"""
import math
from collections import Counter

import numpy as np

from .rerank import id_tokens
from .retrievers import _tokens


class StoreIndex:
    def __init__(self, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.clear()

    def clear(self):
        self.memories, self.ids, self.row, self.lens = [], [], {}, []
        self.total_len = 0
        self.post = {}  # term -> [rows, tfs, rows_array, tfs_array]
        self.id_tokens, self.id_df, self.id_post = [], Counter(), {}
        self.max_ts = float("-inf")
        self.version = 0
        self.epoch = getattr(self, "epoch", 0) + 1  # bumps on rebuild so cached matrices know their rows moved
        self._lens_arr = None
        self._bm25_cache = (None, None, None)

    def append(self, m):
        r = len(self.memories)
        self.memories.append(m)
        self.ids.append(m.id)
        self.row[m.id] = r
        toks = _tokens(m.text)
        self.lens.append(len(toks))
        self.total_len += len(toks)
        for t, c in Counter(toks).items():
            e = self.post.get(t)
            if e is None:
                e = self.post[t] = [[], [], None, None]
            e[0].append(r)
            e[1].append(c)
            e[2] = e[3] = None
        ids = id_tokens(m.text)
        self.id_tokens.append(ids)
        for t in ids:
            self.id_df[t] += 1
            self.id_post.setdefault(t, []).append(r)
        self.max_ts = max(self.max_ts, m.timestamp)
        self.version += 1
        self._lens_arr = None

    def rebuild(self, memories):
        self.clear()
        for m in memories:
            self.append(m)

    def bm25(self, query):
        """BM25 scores of every memory for `query` (same formula and statistics as retrievers.BM25)."""
        key = (query, self.version, self.epoch)
        if self._bm25_cache[0] == key:
            return self._bm25_cache[1]
        n = len(self.memories)
        out = np.zeros(n)
        if n:
            if self._lens_arr is None:
                self._lens_arr = np.asarray(self.lens, dtype=float)
            avgdl = self.total_len / n or 1.0
            for t in set(_tokens(query)):
                e = self.post.get(t)
                if not e:
                    continue
                if e[2] is None:
                    e[2], e[3] = np.asarray(e[0]), np.asarray(e[1], dtype=float)
                rows, tf = e[2], e[3]
                df = len(rows)
                idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
                out[rows] += idf * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * self._lens_arr[rows] / avgdl))
        self._bm25_cache = (key, out, None)
        return out


class IndexedBM25:
    """Drop-in for retrievers.BM25.scores: uses the index when asked about the whole store."""

    def __init__(self, index, base):
        self.index, self.base = index, base

    def scores(self, query, memories):
        if memories is self.index.memories:
            return self.index.bm25(query)
        return self.base.scores(query, memories)


class IndexedDense:
    """Drop-in for retrievers.Dense.scores with a persistent embedding matrix over the whole store."""

    def __init__(self, index, base):
        self.index, self.base = index, base
        self.E, self.E_n, self._epoch = None, 0, None

    def __getattr__(self, name):  # backend, embed, ... come from the wrapped retriever
        return getattr(self.base, name)

    def _sync(self, n):
        ix = self.index
        if self._epoch != ix.epoch:
            self.E, self.E_n, self._epoch = None, 0, ix.epoch
        if self.E_n < n:
            new = self.base.embed([m.text for m in ix.memories[self.E_n:n]])
            if self.E is None or self.E.shape[0] < n:
                grown = np.empty((max(1024, 2 * n), new.shape[1]), dtype=np.float32)
                if self.E is not None:
                    grown[:self.E_n] = self.E[:self.E_n]
                self.E = grown
            self.E[self.E_n:n] = new
            self.E_n = n

    def scores(self, query, memories):
        if memories is not self.index.memories:
            return self.base.scores(query, memories)
        n = len(memories)
        if n == 0:
            return np.zeros(0)
        self._sync(n)
        return self.E[:n] @ self.base.embed([query], is_query=True)[0]
