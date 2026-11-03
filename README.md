# AgentMem

Long-horizon memory reranker for security-investigation agents. No model training: the reranker is prompting and heuristics only.

## Problem

As an agent's history grows over days and weeks, recency or embedding-similarity retrieval surfaces the wrong memories. The reranker's job is to pick the memories that change the agent's next decision.

## Memory schema (fixed)

One atomic record per memory (`agentmem/schema.py`):

| field | meaning |
|---|---|
| `id` | stable id; for benchmarks this matches the gold-evidence id (LoCoMo `dia_id`, LongMemEval `session:turn`) |
| `text` | the content |
| `timestamp` | epoch seconds |
| `source` | `user_message`, `assistant_message`, `tool_call`, `agent_note` |
| `entities` | entity tags (hosts, users, IPs, ticket IDs, people, places) |
| `importance` | write-time prior in [0, 1] (agent-flagged decisions, confirmed facts) |

## Baselines

1. `recency`: newest first
2. `bm25`: lexical
3. `dense`: bge-small-en-v1.5 top-k (falls back to a hashing embedding when `sentence-transformers` is absent; the table says which backend ran)
4. `crossencoder`: dense top-50 reranked by bge-reranker-base
5. Context compression (summarize-then-retrieve): not implemented yet, see Status

## Metrics

- Retrieval: recall@k and MRR of gold evidence ids, with bootstrap 95% CIs over questions, broken down by question type.
- Answer quality (optional, costs money): token F1 against the gold answer. The LongMemEval LLM-judge is not wired in yet.
- LongMemEval types: single-session, multi-session, temporal reasoning, knowledge update, abstention. LoCoMo: per-category breakdown.
- Rule: tune on LongMemEval_s only; LoCoMo is held out.

## Budget

Every API dollar goes in `COSTS.md`. All model calls are cached on disk under `.cache/` keyed by prompt hash. Retrieval-only runs cost nothing.

## Usage

```bash
make test                                   # smoke tests
make bench DATASET=toy RETRIEVER=bm25
make bench DATASET=longmemeval DATA=path/to/longmemeval_s.json RETRIEVER=dense
make bench DATASET=locomo DATA=path/to/locomo10.json RETRIEVER=bm25
```

Results land in `results/<dataset>_<retriever>.md`.

## Status

- Stage 0: done (this file)
- Stage 1: harness, schema, cache, loaders, four baselines, metrics. Loaders verified on the real `locomo10.json` (1986 questions) and `longmemeval_s_cleaned.json` (500 questions, from `xiaowu0162/longmemeval-cleaned`). All four baselines on both datasets are in `results/baselines.md`.
- LongMemEval_s dense/crossencoder numbers are being regenerated after a same-day date bug fix (see the correction note in `results/baselines.md`); earlier claims about temporal-reasoning being the weakest category were an artifact of that bug.
- Stage 2 spike (`results/stage2_spike.md`, `results/stage2_ce.md`): BM25+dense fusion and a cross-encoder score carry the gains; the time-scope, entity and recency heuristics add nothing measurable, and the tuned combination does not beat the cross-encoder alone. LoCoMo numbers in `stage2_ce.md` are a every-4th-question subsample (497 questions).
- LLM time-scope parse (`results/stage2_llmscope.md`): a cheap-model parser roughly doubles scope coverage (to ~12-14% of questions) but does not improve retrieval; only ~58% of scoped questions have a gold turn inside the window, because session timestamps do not line up with when the described events happened. Rejected.
- Gold evidence is turn-level (`has_answer` for LongMemEval, `evidence` dia_ids for LoCoMo). LongMemEval abstention questions have no gold and are excluded from retrieval metrics.
- Not yet: compression baseline, LLM-judge scoring, answer-accuracy numbers (need the paid answerer), reranker (Stage 2), domain data (Stage 3).
- Data lives in `data/` (git-ignored; re-download from the sources above).
