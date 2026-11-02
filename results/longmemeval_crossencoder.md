## longmemeval / crossencoder (backend: BAAI/bge-reranker-base over sentence-transformers:BAAI/bge-small-en-v1.5)

| type | n | recall@10 | MRR |
|---|---|---|---|
| abstention | 30 | n/a | n/a |
| knowledge-update | 72 | 0.958 [0.917, 0.986] | 0.727 [0.649, 0.803] |
| multi-session | 121 | 0.830 [0.783, 0.878] | 0.669 [0.609, 0.733] |
| single-session-assistant | 56 | 0.982 [0.946, 1.000] | 0.782 [0.700, 0.862] |
| single-session-preference | 30 | 0.783 [0.633, 0.911] | 0.596 [0.453, 0.741] |
| single-session-user | 64 | 0.961 [0.906, 1.000] | 0.806 [0.728, 0.875] |
| temporal-reasoning | 127 | 0.590 [0.512, 0.661] | 0.468 [0.395, 0.547] |
| ALL | 500 | 0.818 [0.789, 0.847] | 0.651 [0.617, 0.686] |
