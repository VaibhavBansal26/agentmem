# Stage 3 baselines: security-domain set (k=10, retrieval only)

85 questions over a 3069-record synthetic history (see `agentmem/domain.py`). Abstention questions have no gold evidence and are excluded from retrieval metrics.

| retriever | recall@10 | MRR |
|---|---|---|
| recency | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| bm25 | 0.818 [0.760, 0.877] | 0.859 [0.803, 0.909] |
| dense | 0.714 [0.643, 0.779] | 0.717 [0.640, 0.798] |
| crossencoder | 0.805 [0.740, 0.864] | 0.833 [0.770, 0.893] |

## domain / recency (backend: n/a)

| type | n | recall@10 | MRR |
|---|---|---|---|
| abstention | 8 | n/a | n/a |
| knowledge-update | 18 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| multi-hop | 20 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| single-hop | 18 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| temporal | 21 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| ALL | 85 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |


## domain / bm25 (backend: n/a)

| type | n | recall@10 | MRR |
|---|---|---|---|
| abstention | 8 | n/a | n/a |
| knowledge-update | 18 | 1.000 [1.000, 1.000] | 0.565 [0.481, 0.667] |
| multi-hop | 20 | 0.500 [0.500, 0.500] | 1.000 [1.000, 1.000] |
| single-hop | 18 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| temporal | 21 | 0.810 [0.690, 0.929] | 0.857 [0.714, 0.952] |
| ALL | 85 | 0.818 [0.760, 0.877] | 0.859 [0.803, 0.909] |


## domain / dense (backend: sentence-transformers:BAAI/bge-small-en-v1.5)

| type | n | recall@10 | MRR |
|---|---|---|---|
| abstention | 8 | n/a | n/a |
| knowledge-update | 18 | 1.000 [1.000, 1.000] | 0.435 [0.398, 0.472] |
| multi-hop | 20 | 0.575 [0.500, 0.650] | 1.000 [1.000, 1.000] |
| single-hop | 18 | 1.000 [1.000, 1.000] | 0.894 [0.773, 1.000] |
| temporal | 21 | 0.357 [0.262, 0.452] | 0.539 [0.339, 0.740] |
| ALL | 85 | 0.714 [0.643, 0.779] | 0.717 [0.640, 0.798] |


## domain / crossencoder (backend: BAAI/bge-reranker-base over sentence-transformers:BAAI/bge-small-en-v1.5)

| type | n | recall@10 | MRR |
|---|---|---|---|
| abstention | 8 | n/a | n/a |
| knowledge-update | 18 | 1.000 [1.000, 1.000] | 0.722 [0.583, 0.861] |
| multi-hop | 20 | 0.500 [0.500, 0.500] | 1.000 [1.000, 1.000] |
| single-hop | 18 | 1.000 [1.000, 1.000] | 0.951 [0.854, 1.000] |
| temporal | 21 | 0.762 [0.595, 0.905] | 0.666 [0.483, 0.827] |
| ALL | 85 | 0.805 [0.740, 0.864] | 0.833 [0.770, 0.893] |
