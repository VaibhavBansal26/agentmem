# Does the final pipeline beat the best baseline on the same questions?

Plan and criterion were written before running: `results/compare_PLAN.md`. Retrieval-only (recall@10, MRR).


## LongMemEval_s, odd-indexed half (PRIMARY: weights were fit on the even half) (n=235)

| system | recall@10 | MRR |
|---|---|---|
| pipeline | 0.848 | 0.705 |
| hybrid_ce | 0.863 | 0.719 |
| ce_baseline | 0.888 | 0.679 |
| dense | 0.842 | 0.630 |
| bm25 | 0.774 | 0.657 |

Paired difference (first minus second), mean [95% CI]:

| comparison | recall@10 | MRR |
|---|---|---|
| pipeline - ce_baseline | -0.040 [-0.071, -0.009] | +0.025 [-0.010, +0.061] |
| hybrid_ce - ce_baseline | -0.024 [-0.052, +0.002] | +0.040 [+0.008, +0.072] |
| pipeline - hybrid_ce | -0.016 [-0.035, +0.002] | -0.015 [-0.029, +0.000] |

Per type, pipeline minus ce_baseline:

| type | n | recall@10 | MRR |
|---|---|---|---|
| knowledge-update | 36 | -0.028 [-0.069, +0.000] | +0.025 [-0.049, +0.105] |
| multi-session | 60 | -0.057 [-0.120, +0.004] | +0.020 [-0.058, +0.104] |
| single-session-assistant | 28 | -0.107 [-0.250, +0.000] | -0.046 [-0.153, +0.060] |
| single-session-preference | 15 | -0.100 [-0.267, +0.000] | -0.083 [-0.244, +0.082] |
| single-session-user | 32 | +0.000 [+0.000, +0.000] | +0.028 [-0.030, +0.086] |
| temporal-reasoning | 64 | -0.008 [-0.082, +0.067] | +0.085 [+0.030, +0.149] |

## LongMemEval_s, all questions (reference only: includes the tuning half) (n=470)

| system | recall@10 | MRR |
|---|---|---|
| pipeline | 0.857 | 0.707 |
| hybrid_ce | 0.865 | 0.723 |
| ce_baseline | 0.872 | 0.671 |
| dense | 0.830 | 0.617 |
| bm25 | 0.776 | 0.664 |

Paired difference (first minus second), mean [95% CI]:

| comparison | recall@10 | MRR |
|---|---|---|
| pipeline - ce_baseline | -0.015 [-0.035, +0.006] | +0.036 [+0.011, +0.059] |
| hybrid_ce - ce_baseline | -0.007 [-0.026, +0.013] | +0.052 [+0.029, +0.074] |
| pipeline - hybrid_ce | -0.008 [-0.018, +0.002] | -0.016 [-0.025, -0.007] |

Per type, pipeline minus ce_baseline:

| type | n | recall@10 | MRR |
|---|---|---|---|
| knowledge-update | 72 | -0.007 [-0.035, +0.014] | +0.016 [-0.038, +0.067] |
| multi-session | 121 | -0.068 [-0.110, -0.028] | +0.020 [-0.030, +0.075] |
| single-session-assistant | 56 | -0.036 [-0.107, +0.018] | +0.013 [-0.057, +0.084] |
| single-session-preference | 30 | -0.033 [-0.133, +0.067] | -0.009 [-0.112, +0.089] |
| single-session-user | 64 | +0.016 [+0.000, +0.047] | +0.039 [-0.010, +0.090] |
| temporal-reasoning | 127 | +0.028 [-0.022, +0.078] | +0.080 [+0.032, +0.130] |

## LoCoMo, all questions (held out from the Stage 2 weight fit) (n=1982)

| system | recall@10 | MRR |
|---|---|---|
| pipeline | 0.692 | 0.566 |
| hybrid_ce | 0.688 | 0.562 |
| ce_baseline | 0.662 | 0.533 |
| dense | 0.557 | 0.358 |
| bm25 | 0.542 | 0.365 |

Paired difference (first minus second), mean [95% CI]:

| comparison | recall@10 | MRR |
|---|---|---|
| pipeline - ce_baseline | +0.030 [+0.014, +0.046] | +0.033 [+0.022, +0.045] |
| hybrid_ce - ce_baseline | +0.026 [+0.010, +0.042] | +0.029 [+0.018, +0.041] |
| pipeline - hybrid_ce | +0.004 [+0.001, +0.007] | +0.005 [+0.003, +0.007] |

Per type, pipeline minus ce_baseline:

| type | n | recall@10 | MRR |
|---|---|---|---|
| adversarial | 446 | +0.147 [+0.107, +0.186] | +0.097 [+0.069, +0.125] |
| multi-hop | 282 | -0.073 [-0.101, -0.047] | -0.022 [-0.053, +0.010] |
| open-domain | 92 | -0.043 [-0.109, +0.018] | -0.027 [-0.061, +0.005] |
| single-hop | 841 | +0.020 [-0.004, +0.043] | +0.023 [+0.006, +0.042] |
| temporal | 321 | +0.004 [-0.023, +0.032] | +0.037 [+0.016, +0.060] |

## Verdict against the pre-registered criterion

- LongMemEval_s (odd half): beats the baseline overall (95% CI above 0 on recall@10 or MRR): **False**
- LongMemEval_s (odd half): beats it in at least 2 question types (CI above 0 on either metric): **False** (1 types)
- LoCoMo: significant regression (CI entirely below 0 on either metric): **False**

**Criterion met: False**
