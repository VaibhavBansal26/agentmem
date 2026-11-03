# Stage 3: security-domain set (k=10, retrieval only, no training)

85 questions over a 3069-record synthetic history. Weights are frozen from LongMemEval_s; nothing is tuned on this set. Pool: fused top-30 plus dense and BM25 top-10.

| config | recall@10 | MRR |
|---|---|---|
| dense only | 0.714 [0.643, 0.779] | 0.717 [0.640, 0.798] |
| dense+bm25 (w=1) | 0.799 [0.740, 0.857] | 0.768 [0.698, 0.836] |
| CE only | 0.831 [0.779, 0.883] | 0.839 [0.780, 0.897] |
| frozen LME-tuned, with CE | 0.831 [0.779, 0.883] | 0.820 [0.761, 0.879] |
| frozen minus scope | 0.831 [0.779, 0.883] | 0.813 [0.754, 0.873] |
| frozen minus recency+cue | 0.831 [0.779, 0.883] | 0.820 [0.758, 0.879] |
| frozen + entity 1 (exploratory) | 0.831 [0.779, 0.883] | 0.820 [0.761, 0.879] |
| frozen + entity 2 (exploratory) | 0.831 [0.779, 0.883] | 0.820 [0.761, 0.879] |

Per type, CE only:

| type | n | recall@10 | MRR |
|---|---|---|---|
| abstention | 8 | n/a | n/a |
| knowledge-update | 18 | 1.000 [1.000, 1.000] | 0.722 [0.583, 0.861] |
| multi-hop | 20 | 0.500 [0.500, 0.500] | 1.000 [1.000, 1.000] |
| single-hop | 18 | 1.000 [1.000, 1.000] | 0.951 [0.854, 1.000] |
| temporal | 21 | 0.857 [0.762, 0.952] | 0.690 [0.524, 0.835] |
| ALL | 85 | 0.831 [0.779, 0.883] | 0.839 [0.780, 0.897] |

Per type, frozen LME-tuned, with CE:

| type | n | recall@10 | MRR |
|---|---|---|---|
| abstention | 8 | n/a | n/a |
| knowledge-update | 18 | 1.000 [1.000, 1.000] | 0.565 [0.481, 0.667] |
| multi-hop | 20 | 0.500 [0.500, 0.500] | 1.000 [1.000, 1.000] |
| single-hop | 18 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| temporal | 21 | 0.857 [0.762, 0.952] | 0.712 [0.548, 0.865] |
| ALL | 85 | 0.831 [0.779, 0.883] | 0.820 [0.761, 0.879] |

Per type, frozen + entity 1 (exploratory):

| type | n | recall@10 | MRR |
|---|---|---|---|
| abstention | 8 | n/a | n/a |
| knowledge-update | 18 | 1.000 [1.000, 1.000] | 0.565 [0.481, 0.667] |
| multi-hop | 20 | 0.500 [0.500, 0.500] | 1.000 [1.000, 1.000] |
| single-hop | 18 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| temporal | 21 | 0.857 [0.762, 0.952] | 0.712 [0.548, 0.865] |
| ALL | 85 | 0.831 [0.779, 0.883] | 0.820 [0.761, 0.879] |
