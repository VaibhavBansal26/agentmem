## longmemeval / dense (backend: sentence-transformers:BAAI/bge-small-en-v1.5)

| type | n | recall@10 | MRR |
|---|---|---|---|
| abstention | 30 | n/a | n/a |
| knowledge-update | 72 | 0.905 [0.845, 0.958] | 0.585 [0.508, 0.662] |
| multi-session | 121 | 0.770 [0.718, 0.819] | 0.651 [0.589, 0.713] |
| single-session-assistant | 56 | 0.982 [0.946, 1.000] | 0.689 [0.597, 0.773] |
| single-session-preference | 30 | 0.844 [0.722, 0.944] | 0.595 [0.464, 0.735] |
| single-session-user | 64 | 0.883 [0.805, 0.953] | 0.627 [0.536, 0.719] |
| temporal-reasoning | 127 | 0.572 [0.495, 0.645] | 0.457 [0.386, 0.533] |
| ALL | 500 | 0.783 [0.752, 0.813] | 0.586 [0.553, 0.623] |
