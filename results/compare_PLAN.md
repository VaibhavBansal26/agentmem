# Pre-registered plan: does the final pipeline beat the best baseline?

Written and pushed before any result from `agentmem/compare.py` exists. This is the roadmap's Stage 2 exit criterion made
operational, and I never ran it end to end on identical questions before now.

**Question.** On the same questions, does the final pipeline retrieve better than the strongest baseline from Stage 1
(dense top-50 reranked by bge-reranker-base), and does a hybrid-plus-cross-encoder blend without my heuristics already
account for any gain?

**Systems** (all retrieve k=10 from each question's full haystack): `ce_baseline` (the Stage 1 cross-encoder baseline,
recomputed here), `hybrid_ce` (dense + BM25 + cross-encoder blend, no heuristics, no chaining), `pipeline` (the final
configuration: cross-encoder blend plus the recency bonus gated on recency cues, plus entity chaining), and `dense`,
`bm25` for reference.

**Data and splits.**
- LongMemEval_s: abstention questions are skipped (no gold evidence). The Stage 2 weights were fit on the even-indexed
  questions (loader order), so the **primary** LongMemEval result uses the **odd-indexed half only**. The full set is
  reported for reference and is contaminated by the tuning half.
- LoCoMo: all questions with gold evidence. The Stage 2 weights were never fit on it.

**Metrics.** recall@10 and MRR of the gold evidence turns. This is a retrieval proxy: it does not measure answer accuracy.

**Statistics.** Per-question paired differences, 95% bootstrap CI over questions (2000 resamples, seed 0).

**Criterion** (the roadmap's, made precise). The pipeline **meets** it if (A) on the LongMemEval odd half it beats
`ce_baseline` overall (the CI of the paired difference lies above 0 on recall@10 or on MRR) **or** in at least two question
types (same test, per type), **and** (B) on LoCoMo it does not significantly regress (the CI of the paired difference is not
entirely below 0 on either metric). Otherwise the project is, for the retrieval claim, a negative result, and the README
and blog will say "benchmarked", not "outperformed".

**Sanity check on the harness.** `ce_baseline` should reproduce the Stage 1 numbers in `results/baselines.md`
(LongMemEval_s 0.872 / 0.671, LoCoMo 0.662 / 0.533 over all questions) up to the small difference from skipping
abstention questions. If it does not, the harness is wrong and nothing else is trusted.

**Caveats declared in advance.**
- The chaining parameters were tuned on a separate synthetic set, but the decision to gate the recency bonus on cues was
  made after looking at the LongMemEval and LoCoMo "no harm" checks (every 4th question of each), so neither public set is
  pristine for the final configuration.
- Per-type tests use small samples and will have wide intervals; "at least two types" is a weak test and I will say so.
- A win here would not show an improvement in end-to-end answer accuracy, which I have not measured on these benchmarks.
