# Stage 4: domain questions through the MCP `recall` tool (k=10, no LLM)

85 questions, one `recall` call each, over stdio to a separate server process. A = dense top-k (reranker off); B = Stage 2/3 reranker (cross-encoder, cue-gated entity-scoped recency, entity chaining) with weights frozen from earlier stages.

Tools exposed: `['recall', 'rerank', 'store_memory']`

| variant | recall@10 | MRR | median recall() latency |
|---|---|---|---|
| A (off) | 0.714 [0.643, 0.779] | 0.717 [0.640, 0.798] | 14 ms |
| B (on) | 0.909 [0.864, 0.948] | 0.902 [0.847, 0.952] | 212 ms |

Per type (recall@10 / MRR):

| type | n | A (off) | B (on) |
|---|---|---|---|
| abstention | 8 | n/a / n/a | n/a / n/a |
| knowledge-update | 18 | 1.000 [1.000, 1.000] / 0.435 [0.398, 0.472] | 1.000 [1.000, 1.000] / 0.917 [0.833, 1.000] |
| multi-hop | 20 | 0.575 [0.500, 0.650] / 1.000 [1.000, 1.000] | 0.800 [0.700, 0.900] / 1.000 [1.000, 1.000] |
| single-hop | 18 | 1.000 [1.000, 1.000] / 0.894 [0.773, 1.000] | 1.000 [1.000, 1.000] / 1.000 [1.000, 1.000] |
| temporal | 21 | 0.357 [0.262, 0.452] / 0.539 [0.339, 0.740] | 0.857 [0.762, 0.952] / 0.711 [0.539, 0.872] |
