## longmemeval / crossencoder (backend: BAAI/bge-reranker-base over sentence-transformers:BAAI/bge-small-en-v1.5)

| type | n | recall@10 | MRR |
|---|---|---|---|
| abstention | 30 | n/a | n/a |
| knowledge-update | 72 | 0.965 [0.931, 0.993] | 0.720 [0.645, 0.792] |
| multi-session | 121 | 0.830 [0.783, 0.878] | 0.669 [0.609, 0.733] |
| single-session-assistant | 56 | 0.982 [0.946, 1.000] | 0.782 [0.700, 0.862] |
| single-session-preference | 30 | 0.783 [0.633, 0.911] | 0.596 [0.453, 0.741] |
| single-session-user | 64 | 0.961 [0.906, 1.000] | 0.806 [0.728, 0.875] |
| temporal-reasoning | 127 | 0.787 [0.726, 0.855] | 0.546 [0.479, 0.612] |
| ALL | 500 | 0.872 [0.846, 0.895] | 0.671 [0.639, 0.705] |
