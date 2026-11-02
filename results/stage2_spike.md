# Stage 2 spike (k=10, retrieval only, no training)

Fit on 250 even-indexed LongMemEval_s questions; dev = 250 odd-indexed; LoCoMo (1986 questions) scored once with frozen weights.

Tuned weights: `{'bm25': 0.5, 'scope': 0.5, 'entity': 0.5, 'recency': 0.0, 'recency_cue': 0.5}` (fit objective 0.779, mean of recall@10 and MRR)


## LongMemEval_s dev half

| config | recall@10 | MRR |
|---|---|---|
| dense only | 0.842 [0.805, 0.880] | 0.630 [0.583, 0.680] |
| dense+bm25 (w=1) | 0.870 [0.833, 0.905] | 0.718 [0.669, 0.760] |
| tuned reranker | 0.886 [0.853, 0.919] | 0.701 [0.654, 0.744] |
| tuned minus bm25 | 0.833 [0.795, 0.873] | 0.624 [0.577, 0.674] |
| tuned minus scope | 0.886 [0.853, 0.919] | 0.701 [0.653, 0.744] |
| tuned minus entity | 0.886 [0.853, 0.918] | 0.703 [0.657, 0.747] |
| tuned minus recency_cue | 0.886 [0.853, 0.919] | 0.704 [0.656, 0.748] |

Per type, tuned reranker:

| type | n | recall@10 | MRR |
|---|---|---|---|
| abstention | 15 | n/a | n/a |
| knowledge-update | 36 | 0.944 [0.875, 1.000] | 0.671 [0.563, 0.772] |
| multi-session | 60 | 0.808 [0.738, 0.874] | 0.662 [0.571, 0.750] |
| single-session-assistant | 28 | 0.964 [0.893, 1.000] | 0.767 [0.636, 0.891] |
| single-session-preference | 15 | 0.878 [0.733, 1.000] | 0.666 [0.475, 0.844] |
| single-session-user | 32 | 1.000 [1.000, 1.000] | 0.770 [0.667, 0.870] |
| temporal-reasoning | 64 | 0.835 [0.752, 0.911] | 0.701 [0.603, 0.791] |
| ALL | 250 | 0.886 [0.853, 0.919] | 0.701 [0.654, 0.744] |

## LoCoMo (held out)

| config | recall@10 | MRR |
|---|---|---|
| dense only | 0.557 [0.538, 0.579] | 0.358 [0.341, 0.377] |
| dense+bm25 (w=1) | 0.635 [0.616, 0.655] | 0.448 [0.431, 0.467] |
| tuned reranker | 0.636 [0.618, 0.656] | 0.445 [0.426, 0.464] |
| tuned minus bm25 | 0.554 [0.535, 0.576] | 0.358 [0.341, 0.376] |
| tuned minus scope | 0.633 [0.615, 0.653] | 0.441 [0.423, 0.461] |
| tuned minus entity | 0.647 [0.629, 0.667] | 0.448 [0.429, 0.467] |
| tuned minus recency_cue | 0.637 [0.619, 0.657] | 0.445 [0.427, 0.464] |

Per type, tuned reranker:

| type | n | recall@10 | MRR |
|---|---|---|---|
| adversarial | 446 | 0.589 [0.545, 0.633] | 0.348 [0.314, 0.384] |
| multi-hop | 282 | 0.403 [0.361, 0.446] | 0.373 [0.328, 0.419] |
| open-domain | 96 | 0.378 [0.293, 0.468] | 0.277 [0.200, 0.359] |
| single-hop | 841 | 0.730 [0.700, 0.760] | 0.494 [0.466, 0.522] |
| temporal | 321 | 0.735 [0.688, 0.782] | 0.563 [0.517, 0.609] |
| ALL | 1986 | 0.636 [0.618, 0.656] | 0.445 [0.426, 0.464] |

Candidate-pool recall ceiling on LongMemEval_s: 0.982
