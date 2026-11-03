"""Two training-free, entity-aware signals layered on the base score.

update_bonus: among candidates that share the question's entities and score close to the best
such candidate, boost the newest one (the latest version of a fact, scoped to its entity).

chain_rank: take the top result, collect the rare identifiers it mentions that the question does
not, find other records mentioning them, and insert the ones most relevant to the original
question right after the top result (a second hop).
"""
from collections import Counter

import numpy as np

from .rerank import _time_rank, _z, id_tokens, query_entities, score

DEFAULT = {"w_upd": 0.0, "w_upd_cue": 0.0, "margin": 2.0, "hops": 0, "df_max": 40, "seeds": 1}


def build_full(q, dense, bm25, tok_cache):
    """Per-question arrays over the whole haystack (dense/BM25 scores and identifier tokens)."""
    mems = [m for m in q.memories if m.timestamp <= q.asked_at]
    if len(mems) == len(q.memories):
        tok = tok_cache.setdefault(id(q.memories), [id_tokens(m.text) for m in mems])
    else:
        tok = [id_tokens(m.text) for m in mems]
    return {
        "ids": [m.id for m in mems],
        "dense": np.asarray(dense.scores(q.text, mems), dtype=np.float32),
        "bm25": np.asarray(bm25.scores(q.text, mems), dtype=np.float32),
        "tok": tok,
    }


_DF = {}


def _df(tok):
    """Document frequencies per token list. The cache keeps a reference to each list so its id()
    cannot be recycled for a different list; it holds only the 8 most recent ones."""
    entry = _DF.get(id(tok))
    if entry is None or entry[0] is not tok:
        if len(_DF) >= 8:
            _DF.pop(next(iter(_DF)))
        entry = _DF[id(tok)] = (tok, Counter(t for s in tok for t in s))
    return entry[1]


def update_bonus(F, FULL, base, p):
    if not (p["w_upd"] or (p["w_upd_cue"] and F["cue"])):
        return base
    qent = query_entities(F["query"])
    if not qent:
        return base
    idx = {m: i for i, m in enumerate(FULL["ids"])}
    ent = np.array([len(qent & FULL["tok"][idx[m]]) / len(qent) for m in F["ids"]])
    top = ent.max()
    if top <= 0:
        return base
    cand = ent == top
    grp = cand & (base >= base[cand].max() - p["margin"])
    if grp.sum() < 2:
        return base
    out = base.copy()
    out[grp] += (p["w_upd"] + (p["w_upd_cue"] if F["cue"] else 0.0)) * _time_rank(F["ts"][grp])
    return out


def chain_rank(F, FULL, ranked_ids, p):
    if not p["hops"]:
        return ranked_ids
    seeds = ranked_ids[: p["seeds"]]
    idx = {m: i for i, m in enumerate(FULL["ids"])}
    df, qent = _df(FULL["tok"]), query_entities(F["query"])
    bridge = set()
    for s in seeds:
        bridge |= {t for t in FULL["tok"][idx[s]]
                   if any(c.isdigit() for c in t) and t not in qent and 2 <= df[t] <= p["df_max"]}
    if not bridge:
        return ranked_ids
    cand = [i for i, tok in enumerate(FULL["tok"]) if tok & bridge and FULL["ids"][i] not in seeds]
    if not cand:
        return ranked_ids
    rel = (_z(FULL["dense"]) + _z(FULL["bm25"]))[cand]
    picks = [FULL["ids"][cand[i]] for i in np.argsort(-rel, kind="stable")[: p["hops"]]]
    return seeds + picks + [m for m in ranked_ids if m not in set(seeds) | set(picks)]


def rank(F, FULL, w, p):
    base = update_bonus(F, FULL, score(F, w), p)
    ranked = [F["ids"][i] for i in np.argsort(-base, kind="stable")]
    return chain_rank(F, FULL, ranked, p)
