# Baselines (k=10, retrieval-only, no answerer)

Dense and cross-encoder are not included yet: sentence-transformers is not installed here.

## locomo / recency (backend: n/a)

| type | n | recall@10 | MRR |
|---|---|---|---|
| adversarial | 446 | 0.011 [0.002, 0.022] | 0.002 [0.000, 0.003] |
| multi-hop | 282 | 0.004 [0.000, 0.009] | 0.004 [0.000, 0.012] |
| open-domain | 96 | 0.014 [0.000, 0.038] | 0.004 [0.000, 0.011] |
| single-hop | 841 | 0.012 [0.005, 0.019] | 0.002 [0.001, 0.003] |
| temporal | 321 | 0.009 [0.000, 0.022] | 0.001 [0.000, 0.002] |
| ALL | 1986 | 0.010 [0.006, 0.015] | 0.002 [0.001, 0.003] |
## locomo / bm25 (backend: n/a)

| type | n | recall@10 | MRR |
|---|---|---|---|
| adversarial | 446 | 0.629 [0.584, 0.672] | 0.387 [0.349, 0.423] |
| multi-hop | 282 | 0.216 [0.181, 0.252] | 0.200 [0.163, 0.240] |
| open-domain | 96 | 0.259 [0.179, 0.342] | 0.183 [0.120, 0.251] |
| single-hop | 841 | 0.610 [0.579, 0.641] | 0.411 [0.382, 0.439] |
| temporal | 321 | 0.612 [0.559, 0.659] | 0.415 [0.369, 0.460] |
| ALL | 1986 | 0.542 [0.523, 0.565] | 0.365 [0.349, 0.384] |
## longmemeval / recency (backend: n/a)

| type | n | recall@10 | MRR |
|---|---|---|---|
| abstention | 30 | n/a | n/a |
| knowledge-update | 72 | 0.014 [0.000, 0.035] | 0.003 [0.000, 0.007] |
| multi-session | 121 | 0.008 [0.000, 0.021] | 0.002 [0.000, 0.004] |
| single-session-assistant | 56 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| single-session-preference | 30 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| single-session-user | 64 | 0.047 [0.000, 0.109] | 0.007 [0.000, 0.019] |
| temporal-reasoning | 127 | 0.013 [0.000, 0.033] | 0.003 [0.000, 0.008] |
| ALL | 500 | 0.014 [0.006, 0.024] | 0.003 [0.001, 0.005] |
## longmemeval / bm25 (backend: n/a)

| type | n | recall@10 | MRR |
|---|---|---|---|
| abstention | 30 | n/a | n/a |
| knowledge-update | 72 | 0.903 [0.840, 0.951] | 0.751 [0.674, 0.827] |
| multi-session | 121 | 0.656 [0.584, 0.719] | 0.589 [0.516, 0.664] |
| single-session-assistant | 56 | 0.911 [0.821, 0.982] | 0.785 [0.692, 0.875] |
| single-session-preference | 30 | 0.444 [0.289, 0.611] | 0.201 [0.103, 0.315] |
| single-session-user | 64 | 0.930 [0.859, 0.984] | 0.823 [0.743, 0.899] |
| temporal-reasoning | 127 | 0.561 [0.489, 0.639] | 0.554 [0.472, 0.639] |
| ALL | 500 | 0.722 [0.686, 0.756] | 0.635 [0.596, 0.672] |
