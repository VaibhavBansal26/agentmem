"""Baseline retrievers. Interface: retrieve(query, memories, k, now) -> ranked list of Memory."""
import hashlib
import inspect
import math
import re
import sqlite3
import threading
from pathlib import Path
from collections import Counter

import numpy as np

from .cache import CACHE_DIR


def _tokens(s):
    return re.findall(r"\w+", s.lower())


class Recency:
    name = "recency"

    def retrieve(self, query, memories, k, now=None):
        pool = [m for m in memories if now is None or m.timestamp <= now]
        return sorted(pool, key=lambda m: m.timestamp, reverse=True)[:k]


class BM25:
    name = "bm25"

    def __init__(self, k1=1.5, b=0.75):
        self.k1, self.b = k1, b

    def scores(self, query, memories):
        docs = [_tokens(m.text) for m in memories]
        n = len(docs)
        avgdl = sum(len(d) for d in docs) / max(n, 1) or 1.0
        df = Counter(t for d in docs for t in set(d))
        out = []
        for d in docs:
            tf = Counter(d)
            s = 0.0
            for t in set(_tokens(query)):
                if t not in tf:
                    continue
                idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
                s += idf * tf[t] * (self.k1 + 1) / (
                    tf[t] + self.k1 * (1 - self.b + self.b * len(d) / avgdl)
                )
            out.append(s)
        return out

    def retrieve(self, query, memories, k, now=None):
        pool = [m for m in memories if now is None or m.timestamp <= now]
        ranked = sorted(zip(self.scores(query, pool), pool), key=lambda x: -x[0])
        return [m for _, m in ranked[:k]]


class Dense:
    """bge-small via sentence-transformers when installed, else a hashing embedding.

    The hashing fallback is lexical in spirit, so numbers from it are NOT
    comparable with published dense results. `backend` records which one ran.
    """

    name = "dense"
    QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

    def __init__(self, model="BAAI/bge-small-en-v1.5", dim=512, cache_path=None, embed_fn=None):
        """cache_path: None = <repo cache dir>/embeddings.sqlite, False = no disk cache, else a path.
        embed_fn: optional callable(texts) -> (n, d) array; it replaces the model. If it declares an `is_query`
        parameter it is called as embed_fn(texts, is_query=...), so query and passage embeddings can differ."""
        self.dim = dim
        self.embed_fn = embed_fn
        self._model = None
        if embed_fn is not None:
            self.backend = "custom"
        else:
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(model)
                self.backend = f"sentence-transformers:{model}"
            except ImportError:
                self.backend = "hashing-fallback"
        self._memo = {}
        self._disk = None
        self._lock = threading.RLock()
        if self._model is not None and cache_path is not False:
            path = Path(cache_path) if cache_path else CACHE_DIR / "embeddings.sqlite"
            path.parent.mkdir(parents=True, exist_ok=True)
            self._disk = sqlite3.connect(path, timeout=60, check_same_thread=False)
            self._lock = threading.RLock()  # the connection and the model are shared across threads
            self._disk.execute("CREATE TABLE IF NOT EXISTS emb (h TEXT PRIMARY KEY, v BLOB)")
            self._model_name = model

    def _key(self, text):
        return hashlib.sha1(f"{self._model_name}|{text}".encode()).hexdigest()

    def _hash_embed(self, text):
        v = np.zeros(self.dim, dtype=np.float32)
        for t in _tokens(text):
            h = int(hashlib.md5(t.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0 if (h >> 64) % 2 else -1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed(self, texts, is_query=False):
        with self._lock:
            return self._embed(texts, is_query)

    def _call_embed_fn(self, texts, is_query):
        """Pass is_query only to an embedder that declares a parameter with that name (or takes **kwargs)."""
        try:
            params = inspect.signature(self.embed_fn).parameters
            wants_flag = "is_query" in params or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
        except (TypeError, ValueError):
            wants_flag = False
        vecs = np.asarray(self.embed_fn(texts, is_query=is_query) if wants_flag else self.embed_fn(texts),
                          dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.where(norms == 0, 1.0, norms)

    def _embed(self, texts, is_query=False):
        if self.embed_fn is not None:
            missing = [t for t in dict.fromkeys(texts) if (is_query, t) not in self._memo]
            if missing:
                self._memo.update({(is_query, t): v for t, v in zip(missing, self._call_embed_fn(missing, is_query))})
            return np.stack([self._memo[(is_query, t)] for t in texts])
        if self._model is None:
            return np.stack([self._hash_embed(t) for t in texts])
        if is_query:
            texts = [self.QUERY_PREFIX + t for t in texts]
        missing = [t for t in dict.fromkeys(texts) if t not in self._memo]
        if missing and self._disk is not None:  # on-disk cache shared across processes
            keys = {self._key(t): t for t in missing}
            found = {}
            for i in range(0, len(keys), 500):
                chunk = list(keys)[i:i + 500]
                q = f"SELECT h, v FROM emb WHERE h IN ({','.join('?' * len(chunk))})"
                found.update({h: np.frombuffer(v, dtype=np.float32) for h, v in self._disk.execute(q, chunk)})
            self._memo.update({keys[h]: v for h, v in found.items()})
            missing = [t for t in missing if t not in self._memo]
        if missing:
            vecs = self._model.encode(missing, normalize_embeddings=True, batch_size=64)
            self._memo.update(zip(missing, vecs))
            if self._disk is not None:
                self._disk.executemany("INSERT OR REPLACE INTO emb VALUES (?, ?)", [
                    (self._key(t), np.asarray(v, dtype=np.float32).tobytes()) for t, v in zip(missing, vecs)])
                self._disk.commit()
        return np.stack([self._memo[t] for t in texts])

    def scores(self, query, memories):
        d = self.embed([m.text for m in memories])
        q = self.embed([query], is_query=True)[0]
        return d @ q

    def retrieve(self, query, memories, k, now=None):
        pool = [m for m in memories if now is None or m.timestamp <= now]
        if not pool:
            return []
        s = self.scores(query, pool)
        return [pool[i] for i in np.argsort(-s)[:k]]


class CrossEncoder:
    """Dense top-50, then bge-reranker-base over the candidates."""

    name = "crossencoder"

    def __init__(self, model="BAAI/bge-reranker-base", candidates=50):
        try:
            from sentence_transformers import CrossEncoder as CE
        except ImportError as e:
            raise RuntimeError("pip install sentence-transformers for this baseline") from e
        self._ce = CE(model)
        self._first = Dense()
        self.candidates = candidates
        self.backend = f"{model} over {self._first.backend}"

    def retrieve(self, query, memories, k, now=None):
        cands = self._first.retrieve(query, memories, self.candidates, now)
        if not cands:
            return []
        s = self._ce.predict([(query, m.text) for m in cands])
        return [cands[i] for i in np.argsort(-np.asarray(s))[:k]]


REGISTRY = {
    "recency": Recency,
    "bm25": BM25,
    "dense": Dense,
    "crossencoder": CrossEncoder,
}


def get(name):
    if name not in REGISTRY:
        raise SystemExit(f"unknown retriever {name!r}; choose from {sorted(REGISTRY)}")
    return REGISTRY[name]()
