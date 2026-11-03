"""Training-free reranker: a weighted sum of cheap signals over a hybrid candidate pool."""
import re
from datetime import datetime, timezone

import numpy as np

from .retrievers import BM25, Dense, _tokens

DAY = 86400.0
UNIT_DAYS = {"day": 1, "week": 7, "month": 30, "year": 365}
COUNT = {
    "an": 1, "a": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "a couple of": 2, "a few": 3,
}
MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august",
          "september", "october", "november", "december"]
COUNT_RE = "|".join(sorted(COUNT, key=len, reverse=True))
AGO_RE = re.compile(rf"\b(\d+|{COUNT_RE})\s+(day|week|month|year)s?\s+ago")
MONTH_RE = re.compile(rf"\bin ({'|'.join(MONTHS)})(?:\s+(\d{{4}}))?")
CUE_RE = re.compile(r"\b(current|currently|now|latest|most recent|still|nowadays|these days|updated?)\b")
QUESTION_WORDS = {"what", "which", "who", "whom", "when", "where", "why", "how", "did", "do",
                  "does", "is", "are", "was", "were", "has", "have", "had", "can", "could",
                  "would", "should", "the", "i", "my"}


def time_scope(query, now):
    """Heuristic (start, end) epoch window implied by the query, or None."""
    q = query.lower()
    m = AGO_RE.search(q)
    if m:
        raw, unit = m.group(1), m.group(2)
        n = int(raw) if raw.isdigit() else COUNT[raw]
        center = now - n * UNIT_DAYS[unit] * DAY
        tol = max(DAY, 0.25 * n * UNIT_DAYS[unit] * DAY)
        return center - tol, center + tol
    m = MONTH_RE.search(q)
    if m:
        month = MONTHS.index(m.group(1)) + 1
        cur = datetime.fromtimestamp(now, timezone.utc)
        year = int(m.group(2)) if m.group(2) else (cur.year if month <= cur.month else cur.year - 1)
        start = datetime(year, month, 1, tzinfo=timezone.utc).timestamp()
        end = datetime(year + (month == 12), month % 12 + 1, 1, tzinfo=timezone.utc).timestamp()
        return start, end
    for phrase, days in (("yesterday", 2), ("last week", 14), ("last month", 62),
                         ("last year", 730), ("this week", 7), ("today", 1)):
        if phrase in q:
            return now - days * DAY, now
    return None


_ID_RE = re.compile(r"[a-z0-9][\w.:/-]*")


def id_tokens(text):
    """Identifier-style tokens (host-07, 198.51.100.5, T-4003) kept whole, as in query_entities."""
    return {t.strip(".:-/") for t in _ID_RE.findall(text.lower())}


def query_entities(query):
    """Lowercased proper-noun-ish and numeric tokens from the query."""
    ents = set()
    for i, tok in enumerate(re.findall(r"[A-Za-z0-9][\w.:/-]*", query)):
        low = tok.lower().strip(".:-/")
        if low in QUESTION_WORDS:
            continue
        if any(c.isdigit() for c in tok) or (tok[0].isupper() and i > 0):
            ents.add(low)
    return ents


def featurize(q, dense, bm25, n_dense=50, n_bm25=20):
    pool = [i for i, m in enumerate(q.memories) if m.timestamp <= q.asked_at]
    mems = [q.memories[i] for i in pool]
    d = np.asarray(dense.scores(q.text, mems))
    b = np.asarray(bm25.scores(q.text, mems))
    cand = sorted(set(np.argsort(-d)[:n_dense]) | set(np.argsort(-b)[:n_bm25]))
    scope = time_scope(q.text, q.asked_at)
    ents = query_entities(q.text)
    ts = np.array([mems[i].timestamp for i in cand])
    ent = np.array([
        len(ents & id_tokens(mems[i].text)) / len(ents) if ents else 0.0 for i in cand
    ])
    in_scope = (
        ((ts >= scope[0]) & (ts <= scope[1])).astype(float) if scope else np.zeros(len(cand))
    )
    return {
        "id": q.id, "query": q.text, "qtype": q.qtype, "gold": set(q.gold_ids),
        "ids": [mems[i].id for i in cand],
        "dense": d[cand], "bm25": b[cand], "ts": ts, "entity": ent,
        "in_scope": in_scope, "has_scope": scope is not None,
        "cue": bool(CUE_RE.search(q.text.lower())),
    }


def _z(x):
    return (x - x.mean()) / (x.std() + 1e-9)


def _time_rank(ts):
    order = ts.argsort().argsort()
    return order / max(len(ts) - 1, 1)


WEIGHTS = {"dense": 1.0, "ce": 0.0, "scope_llm": 0.0, "bm25": 0.0, "scope": 0.0, "entity": 0.0, "recency": 0.0, "recency_cue": 0.0}


def score(F, w):
    s = w["dense"] * _z(F["dense"]) + w["bm25"] * _z(F["bm25"])
    if w["ce"] and "ce" in F:
        s = s + w["ce"] * _z(F["ce"])
    s = s + w["scope"] * F["in_scope"] + w["entity"] * F["entity"]
    if w["scope_llm"] and "in_scope_llm" in F:
        s = s + w["scope_llm"] * F["in_scope_llm"]
    tr = _time_rank(F["ts"])
    s = s + w["recency"] * tr + (w["recency_cue"] * tr if F["cue"] else 0.0)
    return s


def rank_ids(F, w):
    return [F["ids"][i] for i in np.argsort(-score(F, w), kind="stable")]


def with_ce_scores(F, texts, ce, top_fused=30, top_each=10, batch_size=32):
    """Restrict F to a small pool and add cross-encoder scores for it.

    Pool = fused (dense+bm25) top-30 plus the dense and BM25 top-10, so the dense-only
    and hybrid top-10 rankings are unchanged by the restriction.
    """
    fused = _z(F["dense"]) + _z(F["bm25"])
    keep = set(np.argsort(-fused)[:top_fused]) | set(np.argsort(-F["dense"])[:top_each]) \
        | set(np.argsort(-F["bm25"])[:top_each])
    keep = sorted(keep)
    out = {k: (v[keep] if isinstance(v, np.ndarray) else v) for k, v in F.items()}
    out["ids"] = [F["ids"][i] for i in keep]
    out["ce"] = np.asarray(ce.predict([(F["query"], texts[F["ids"][i]]) for i in keep],
                                      batch_size=batch_size, show_progress_bar=False))
    return out
