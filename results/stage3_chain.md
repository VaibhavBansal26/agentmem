# Entity-scoped recency and entity chaining (k=10, retrieval only, no training)

Base scorer = the Stage 2 weights frozen from LongMemEval_s. The two new signals' parameters are tuned ONLY on a second synthetic set (seed 11, 86 questions); evaluation sets were never used for tuning. Domain sets share a generator, so this is in-distribution; LoCoMo and LongMemEval_s are no-harm checks.

Tuned parameters: `{'w_upd': 4, 'w_upd_cue': 0.0, 'margin': 4, 'hops': 3, 'df_max': 40, 'seeds': 1}` (dev objective 0.880, base 0.814)


## Domain test set (seed 7), n=85

| config | recall@10 | MRR |
|---|---|---|
| base (frozen Stage 2) | 0.831 [0.779, 0.883] | 0.820 [0.761, 0.879] |
| + entity-scoped recency | 0.831 [0.779, 0.883] | 0.895 [0.838, 0.950] |
| + entity chaining | 0.909 [0.864, 0.948] | 0.819 [0.758, 0.879] |
| + both | 0.909 [0.864, 0.948] | 0.894 [0.834, 0.949] |

Per type, base (frozen Stage 2):

| type | n | recall@10 | MRR |
|---|---|---|---|
| abstention | 8 | n/a | n/a |
| knowledge-update | 18 | 1.000 [1.000, 1.000] | 0.565 [0.481, 0.667] |
| multi-hop | 20 | 0.500 [0.500, 0.500] | 1.000 [1.000, 1.000] |
| single-hop | 18 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| temporal | 21 | 0.857 [0.762, 0.952] | 0.712 [0.548, 0.865] |
| ALL | 85 | 0.831 [0.779, 0.883] | 0.820 [0.761, 0.879] |

Per type, + both:

| type | n | recall@10 | MRR |
|---|---|---|---|
| abstention | 8 | n/a | n/a |
| knowledge-update | 18 | 1.000 [1.000, 1.000] | 0.917 [0.833, 1.000] |
| multi-hop | 20 | 0.800 [0.700, 0.900] | 1.000 [1.000, 1.000] |
| single-hop | 18 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| temporal | 21 | 0.857 [0.762, 0.952] | 0.682 [0.514, 0.849] |
| ALL | 85 | 0.909 [0.864, 0.948] | 0.894 [0.834, 0.949] |

## LoCoMo (every 4th question; no-harm check), n=497

| config | recall@10 | MRR |
|---|---|---|
| base (frozen Stage 2) | 0.711 [0.672, 0.746] | 0.583 [0.545, 0.619] |
| + entity-scoped recency | 0.711 [0.672, 0.746] | 0.568 [0.531, 0.605] |
| + entity chaining | 0.711 [0.672, 0.746] | 0.583 [0.546, 0.621] |
| + both | 0.711 [0.672, 0.746] | 0.569 [0.532, 0.605] |

Per type, base (frozen Stage 2):

| type | n | recall@10 | MRR |
|---|---|---|---|
| adversarial | 113 | 0.730 [0.646, 0.801] | 0.510 [0.430, 0.580] |
| multi-hop | 76 | 0.478 [0.405, 0.554] | 0.547 [0.455, 0.643] |
| open-domain | 20 | 0.360 [0.158, 0.561] | 0.212 [0.069, 0.382] |
| single-hop | 210 | 0.780 [0.722, 0.837] | 0.614 [0.555, 0.670] |
| temporal | 78 | 0.808 [0.724, 0.885] | 0.730 [0.641, 0.817] |
| ALL | 497 | 0.711 [0.672, 0.746] | 0.583 [0.545, 0.619] |

Per type, + both:

| type | n | recall@10 | MRR |
|---|---|---|---|
| adversarial | 113 | 0.730 [0.646, 0.801] | 0.527 [0.448, 0.599] |
| multi-hop | 76 | 0.478 [0.405, 0.554] | 0.513 [0.422, 0.609] |
| open-domain | 20 | 0.360 [0.158, 0.561] | 0.212 [0.069, 0.382] |
| single-hop | 210 | 0.780 [0.722, 0.837] | 0.601 [0.545, 0.658] |
| temporal | 78 | 0.808 [0.724, 0.885] | 0.681 [0.599, 0.765] |
| ALL | 497 | 0.711 [0.672, 0.746] | 0.569 [0.532, 0.605] |

## LongMemEval_s (every 4th question; no-harm check), n=125

| config | recall@10 | MRR |
|---|---|---|
| base (frozen Stage 2) | 0.877 [0.831, 0.919] | 0.724 [0.661, 0.784] |
| + entity-scoped recency | 0.877 [0.831, 0.919] | 0.678 [0.612, 0.740] |
| + entity chaining | 0.878 [0.831, 0.921] | 0.708 [0.641, 0.773] |
| + both | 0.878 [0.831, 0.921] | 0.662 [0.593, 0.726] |

Per type, base (frozen Stage 2):

| type | n | recall@10 | MRR |
|---|---|---|---|
| abstention | 9 | n/a | n/a |
| knowledge-update | 17 | 0.971 [0.912, 1.000] | 0.703 [0.553, 0.853] |
| multi-session | 30 | 0.749 [0.649, 0.850] | 0.702 [0.579, 0.819] |
| single-session-assistant | 14 | 1.000 [1.000, 1.000] | 0.728 [0.546, 0.901] |
| single-session-preference | 8 | 0.771 [0.500, 0.958] | 0.713 [0.413, 1.000] |
| single-session-user | 16 | 0.969 [0.906, 1.000] | 0.837 [0.674, 0.969] |
| temporal-reasoning | 31 | 0.874 [0.770, 0.958] | 0.701 [0.566, 0.823] |
| ALL | 125 | 0.877 [0.831, 0.919] | 0.724 [0.661, 0.784] |

Per type, + both:

| type | n | recall@10 | MRR |
|---|---|---|---|
| abstention | 9 | n/a | n/a |
| knowledge-update | 17 | 0.971 [0.912, 1.000] | 0.654 [0.503, 0.804] |
| multi-session | 30 | 0.763 [0.667, 0.858] | 0.636 [0.517, 0.761] |
| single-session-assistant | 14 | 1.000 [1.000, 1.000] | 0.692 [0.514, 0.869] |
| single-session-preference | 8 | 0.771 [0.500, 0.958] | 0.775 [0.475, 1.000] |
| single-session-user | 16 | 0.969 [0.906, 1.000] | 0.756 [0.583, 0.906] |
| temporal-reasoning | 31 | 0.863 [0.754, 0.953] | 0.598 [0.461, 0.732] |
| ALL | 125 | 0.878 [0.831, 0.921] | 0.662 [0.593, 0.726] |
