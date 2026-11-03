## domain / crossencoder (backend: BAAI/bge-reranker-base over sentence-transformers:BAAI/bge-small-en-v1.5)

| type | n | recall@10 | MRR |
|---|---|---|---|
| abstention | 8 | n/a | n/a |
| knowledge-update | 18 | 1.000 [1.000, 1.000] | 0.722 [0.583, 0.861] |
| multi-hop | 20 | 0.500 [0.500, 0.500] | 1.000 [1.000, 1.000] |
| single-hop | 18 | 1.000 [1.000, 1.000] | 0.951 [0.854, 1.000] |
| temporal | 21 | 0.762 [0.595, 0.905] | 0.666 [0.483, 0.827] |
| ALL | 85 | 0.805 [0.740, 0.864] | 0.833 [0.770, 0.893] |
