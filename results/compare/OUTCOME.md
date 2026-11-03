# Outcome of the pre-registered comparison (plan: `results/compare_PLAN.md`; numbers: `report.md`; raw rows: `*.jsonl`)

**Verdict: the criterion is NOT met.** The pipeline does not beat the best baseline (dense top-50 + bge-reranker-base) on
LongMemEval_s by the pre-registered test, so for the public-benchmark retrieval claim the right word is "benchmarked", not
"outperformed". The harness was sound: its recomputed baseline reproduces the Stage 1 numbers exactly
(LongMemEval_s 0.872 / 0.671).

| | pipeline vs ce_baseline, recall@10 | MRR |
|---|---|---|
| LongMemEval_s, odd half (235 q, not used for tuning) | -0.040 [-0.071, -0.009] (significantly worse) | +0.025 [-0.010, +0.061] (not significant) |
| LoCoMo, all (1,982 q, held out from the weight fit) | +0.030 [+0.014, +0.046] | +0.033 [+0.022, +0.045] |

What it means:
- The part that helps is the **hybrid dense + BM25 + cross-encoder blend** (`hybrid_ce`): +0.026 recall and +0.029 MRR over the
  standard baseline on LoCoMo, and +0.040 MRR (significant) on the LongMemEval odd half, at a small recall cost there.
- **My heuristics add almost nothing:** pipeline minus hybrid_ce is +0.004 / +0.005 on LoCoMo and slightly negative on
  LongMemEval (-0.016 recall, -0.015 MRR on the odd half, borderline).
- **The LoCoMo average hides a split.** Adversarial questions gain a lot (recall +0.147) while multi-hop recall is
  significantly worse (-0.073 [-0.101, -0.047]). I suspect lexical fusion explains the adversarial gain (BM25 beat dense there in
  Stage 1), but I have not tested it.

Declared caveats (see the plan): retrieval-only (no answer accuracy); neither public set is pristine for the final
configuration, because the recency-cue gate was chosen after looking at every-4th-question subsets of both; per-type tests
are small-sample. **Untested hypothesis:** the pipeline scores about 40 candidates against the baseline's 50, which could
cap recall; widening the pool is a possible follow-up and would be exploratory, outside this verdict.
