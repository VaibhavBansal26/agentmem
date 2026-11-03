"""Public API: `from memorank import Memory`.

    mem = Memory("memories.db")
    mem.add("Host web-12 role set to production web server", entities=["web-12"])
    for hit in mem.recall("What is the current role of web-12?", k=5):
        print(hit.rank, hit.text)

Retrieval is dense + BM25 candidates, then (when available) a cross-encoder score, a cue-gated entity-scoped
recency bonus and entity chaining. Everything heavy is optional: with no extras installed it still works, using
a lexical hashing embedding and no cross-encoder (lower quality; a warning says so).
"""
import importlib.util
import warnings
from dataclasses import dataclass, field
from datetime import date, datetime, timezone

from .service import MemoryService


@dataclass(frozen=True)
class Hit:
    id: str
    text: str
    timestamp: float
    source: str
    entities: tuple
    rank: int

    @property
    def time(self):
        return datetime.fromtimestamp(self.timestamp, timezone.utc)


@dataclass
class Config:
    """Everything tunable. None means the evaluated defaults (see README for how they were chosen)."""

    weights: dict = None  # signal weights, see memorank.service.WEIGHTS
    chain: dict = None  # entity-chaining and recency-bonus parameters, see memorank.service.CHAIN
    dense_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model: str = "BAAI/bge-reranker-base"


def _ts(value):
    """Accept epoch seconds, datetime (naive = UTC), date, or an ISO string."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if isinstance(value, date) and not isinstance(value, datetime):
        value = datetime(value.year, value.month, value.day)
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.timestamp()


def _day(value):
    """Window bounds are date-granular: any accepted time value becomes 'YYYY-MM-DD' (UTC)."""
    if value is None:
        return None
    return datetime.fromtimestamp(_ts(value), timezone.utc).strftime("%Y-%m-%d")


class Memory:
    """A persistent, thread-safe memory with reranked recall.

    path: SQLite file, or ":memory:". embedder: optional callable(texts) -> (n, d) array (add an `is_query=False`
    parameter if queries and passages embed differently),
    replaces the dense model. scorer: optional callable(list of (query, text)) -> list of floats, replaces the
    cross-encoder (higher = more relevant); it is one signal in a weighted blend with dense and BM25 scores,
    not an override. rerank=False gives plain dense top-k. cross_encoder=False skips the cross-encoder
    (much faster, lower quality) even when sentence-transformers is installed. now: fix the clock (epoch)
    to replay a historical store; by default memories newer than the current time are not recalled.
    """

    def __init__(self, path=":memory:", *, embedder=None, scorer=None, rerank=True, cross_encoder=True,
                 config=None, now=None):
        cfg = config or Config()
        have_st = importlib.util.find_spec("sentence_transformers") is not None
        if embedder is None and not have_st:
            warnings.warn("sentence-transformers is not installed: dense retrieval falls back to a lexical hashing "
                          "embedding and there is no cross-encoder. Quality is lower; "
                          "pip install 'memorank[dense]' for the evaluated pipeline.", stacklevel=2)
        cache = False if path == ":memory:" else f"{path}.emb.sqlite"  # never write into the caller's cwd
        self._svc = MemoryService(
            path, rerank=rerank, use_ce=cross_encoder and scorer is None and have_st, now=_ts(now), scorer=scorer,
            weights=cfg.weights, chain_params=cfg.chain, ce_model=cfg.reranker_model,
            dense_kwargs={"model": cfg.dense_model, "cache_path": cache, "embed_fn": embedder})

    # --- writing -----------------------------------------------------------------------------
    def add(self, text, *, source="agent_note", entities=(), importance=0.0, timestamp=None, id=None):
        """Store one memory and return its id. entities are optional tags; identifiers in the text
        (hosts, IPs, ticket ids, proper nouns) are picked up automatically at recall time."""
        return self._svc.store(text, source, tuple(entities), importance, _ts(timestamp), id)

    def add_many(self, records):
        """records: iterable of dicts with 'text' and optional source/entities/importance/timestamp/id."""
        from .schema import Memory as Record
        import uuid

        rows = []
        for r in records:
            rows.append(Record(r.get("id") or f"m{uuid.uuid4().hex[:10]}", r["text"],
                               _ts(r.get("timestamp")) if r.get("timestamp") is not None else self._svc.clock(),
                               r.get("source", "agent_note"), tuple(r.get("entities", ())),
                               r.get("importance", 0.0)))
        self._svc.bulk_load(rows)
        return [r.id for r in rows]

    def delete(self, id):
        return self._svc.delete(id)

    # --- reading -----------------------------------------------------------------------------
    def recall(self, query, k=8, *, start=None, end=None, rerank=None):
        """Best-first hits for a question. start/end (date-granular) restrict to a time window."""
        out = self._svc.recall(query, k, _day(start), _day(end), rerank)
        hits = []
        for o in out:
            m = self._svc.get(o["id"])
            hits.append(Hit(m.id, m.text, m.timestamp, m.source, m.entities, o["rank"]))
        return hits

    def rerank(self, query, candidates, k=None):
        """Rerank caller-supplied candidates: dicts with 'text' and optional 'timestamp' (ISO) and 'id'.
        Returns dicts in best-first order. This is the 'bring your own store' entry point."""
        return self._svc.rerank_candidates(query, candidates, k)

    def get(self, id):
        m = self._svc.get(id)
        return Hit(m.id, m.text, m.timestamp, m.source, m.entities, 0) if m else None

    def __len__(self):
        return len(self._svc.memories)

    def close(self):
        self._svc.db.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
