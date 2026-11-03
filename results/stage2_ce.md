# Stage 2 spike with cross-encoder feature (k=10, retrieval only, no training)

Pool per question: fused (dense+BM25) top-30 plus dense and BM25 top-10 (same pool for every row). Weights fit on even-indexed LongMemEval_s only; dev = odd-indexed; LoCoMo scored once with frozen weights.

Tuned without CE: `{'dense': 1.0, 'ce': 0.0, 'bm25': 1, 'scope': 2, 'entity': 0.0, 'recency': 0.0, 'recency_cue': 0.0}`

Tuned with CE: `{'dense': 1.0, 'ce': 2, 'bm25': 2, 'scope': 4, 'entity': 0.0, 'recency': 0.5, 'recency_cue': 0.5}` (fit objective 0.796)


## LongMemEval_s dev half

| config | recall@10 | MRR |
|---|---|---|
| dense only | 0.842 [0.805, 0.880] | 0.630 [0.583, 0.680] |
| dense+bm25 (w=1) | 0.857 [0.817, 0.895] | 0.712 [0.662, 0.755] |
| CE only | 0.895 [0.860, 0.928] | 0.693 [0.647, 0.739] |
| tuned, no CE | 0.861 [0.824, 0.899] | 0.714 [0.664, 0.756] |
| tuned, with CE | 0.871 [0.833, 0.908] | 0.722 [0.676, 0.767] |
| with CE, minus ce | 0.833 [0.789, 0.872] | 0.695 [0.645, 0.741] |
| with CE, minus bm25 | 0.887 [0.857, 0.921] | 0.701 [0.655, 0.746] |
| with CE, minus scope | 0.863 [0.825, 0.901] | 0.721 [0.675, 0.766] |
| with CE, minus recency | 0.869 [0.831, 0.907] | 0.720 [0.673, 0.764] |
| with CE, minus recency_cue | 0.873 [0.835, 0.909] | 0.725 [0.679, 0.770] |

Per type, tuned with CE:

| type | n | recall@10 | MRR |
|---|---|---|---|
| abstention | 15 | n/a | n/a |
| knowledge-update | 36 | 0.944 [0.889, 0.986] | 0.745 [0.633, 0.852] |
| multi-session | 60 | 0.794 [0.719, 0.865] | 0.704 [0.621, 0.787] |
| single-session-assistant | 28 | 0.893 [0.786, 1.000] | 0.780 [0.643, 0.902] |
| single-session-preference | 15 | 0.811 [0.611, 0.967] | 0.611 [0.433, 0.800] |
| single-session-user | 32 | 1.000 [1.000, 1.000] | 0.878 [0.784, 0.953] |
| temporal-reasoning | 64 | 0.843 [0.767, 0.916] | 0.650 [0.559, 0.741] |
| ALL | 250 | 0.871 [0.833, 0.908] | 0.722 [0.676, 0.767] |

## LoCoMo (held out)

| config | recall@10 | MRR |
|---|---|---|
| dense only | 0.550 [0.510, 0.589] | 0.367 [0.331, 0.403] |
| dense+bm25 (w=1) | 0.653 [0.613, 0.692] | 0.470 [0.434, 0.507] |
| CE only | 0.728 [0.692, 0.763] | 0.572 [0.536, 0.610] |
| tuned, no CE | 0.660 [0.622, 0.698] | 0.472 [0.437, 0.509] |
| tuned, with CE | 0.711 [0.672, 0.746] | 0.583 [0.545, 0.619] |
| with CE, minus ce | 0.637 [0.599, 0.676] | 0.457 [0.422, 0.493] |
| with CE, minus bm25 | 0.717 [0.681, 0.752] | 0.574 [0.537, 0.613] |
| with CE, minus scope | 0.708 [0.669, 0.743] | 0.580 [0.543, 0.617] |
| with CE, minus recency | 0.709 [0.671, 0.744] | 0.581 [0.542, 0.619] |
| with CE, minus recency_cue | 0.711 [0.672, 0.746] | 0.583 [0.545, 0.619] |

Per type, tuned with CE:

| type | n | recall@10 | MRR |
|---|---|---|---|
| adversarial | 113 | 0.730 [0.646, 0.801] | 0.510 [0.430, 0.580] |
| multi-hop | 76 | 0.478 [0.405, 0.554] | 0.547 [0.455, 0.643] |
| open-domain | 20 | 0.360 [0.158, 0.561] | 0.212 [0.069, 0.382] |
| single-hop | 210 | 0.780 [0.722, 0.837] | 0.614 [0.555, 0.670] |
| temporal | 78 | 0.808 [0.724, 0.885] | 0.730 [0.641, 0.817] |
| ALL | 497 | 0.711 [0.672, 0.746] | 0.583 [0.545, 0.619] |
