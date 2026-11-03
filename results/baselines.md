# Baselines (k=10, retrieval only, no answerer)

Summary (recall@10 / MRR over all scored questions; LongMemEval abstention questions have no gold evidence and are excluded):

| retriever | LoCoMo (1,986 q) | LongMemEval_s (500 q) |
|---|---|---|
| recency | 0.010 / 0.002 | 0.014 / 0.003 |
| bm25 | 0.542 / 0.365 | 0.776 / 0.664 |
| dense (bge-small-en-v1.5) | 0.557 / 0.358 | 0.830 / 0.617 |
| crossencoder (bge-reranker-base over dense top-50) | 0.662 / 0.533 | 0.872 / 0.671 |

All four LongMemEval rows were regenerated after fixing a harness bug (see below). LoCoMo is unaffected by that bug and
those rows were not rerun.

**History of the fix.** The first LongMemEval run dropped turns dated after the question's clock time, but some questions
are dated earlier on the same day as their sessions. That removed 63 of 253 gold turns for temporal-reasoning (39 of 127
questions lost at least one). Fixed in `loaders.py` (a question date now means the whole day), with a regression test.
Effect on the cross-encoder row: recall@10 0.818 to 0.872, MRR 0.651 to 0.671; temporal-reasoning recall 0.590 to 0.787.

**What the corrected numbers say.** The earlier conclusion that temporal reasoning is the weakest category was an artifact
of the bug. After the fix, the weakest categories by recall@10 are single-session-preference and multi-session for most
retrievers (BM25: 0.444 and 0.656), while temporal-reasoning keeps the lowest MRR for the cross-encoder (0.546).

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


## locomo / dense (backend: sentence-transformers:BAAI/bge-small-en-v1.5)

| type | n | recall@10 | MRR |
|---|---|---|---|
| adversarial | 446 | 0.457 [0.413, 0.506] | 0.225 [0.194, 0.258] |
| multi-hop | 282 | 0.379 [0.338, 0.424] | 0.336 [0.293, 0.382] |
| open-domain | 96 | 0.339 [0.254, 0.429] | 0.238 [0.169, 0.315] |
| single-hop | 841 | 0.649 [0.619, 0.682] | 0.400 [0.372, 0.427] |
| temporal | 321 | 0.675 [0.626, 0.725] | 0.489 [0.441, 0.536] |
| ALL | 1986 | 0.557 [0.538, 0.579] | 0.358 [0.341, 0.377] |


## locomo / crossencoder (backend: BAAI/bge-reranker-base over sentence-transformers:BAAI/bge-small-en-v1.5)

| type | n | recall@10 | MRR |
|---|---|---|---|
| adversarial | 446 | 0.573 [0.530, 0.618] | 0.393 [0.354, 0.431] |
| multi-hop | 282 | 0.474 [0.432, 0.519] | 0.501 [0.452, 0.549] |
| open-domain | 96 | 0.450 [0.362, 0.541] | 0.339 [0.255, 0.417] |
| single-hop | 841 | 0.761 [0.731, 0.788] | 0.601 [0.571, 0.629] |
| temporal | 321 | 0.753 [0.705, 0.795] | 0.633 [0.584, 0.679] |
| ALL | 1986 | 0.662 [0.643, 0.682] | 0.533 [0.514, 0.552] |


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
| knowledge-update | 72 | 0.910 [0.854, 0.958] | 0.751 [0.674, 0.827] |
| multi-session | 121 | 0.656 [0.584, 0.719] | 0.589 [0.516, 0.664] |
| single-session-assistant | 56 | 0.911 [0.821, 0.982] | 0.785 [0.692, 0.875] |
| single-session-preference | 30 | 0.444 [0.289, 0.611] | 0.201 [0.103, 0.315] |
| single-session-user | 64 | 0.930 [0.859, 0.984] | 0.823 [0.743, 0.899] |
| temporal-reasoning | 127 | 0.757 [0.688, 0.826] | 0.661 [0.591, 0.734] |
| ALL | 500 | 0.776 [0.743, 0.805] | 0.664 [0.626, 0.698] |


## longmemeval / dense (backend: sentence-transformers:BAAI/bge-small-en-v1.5)

| type | n | recall@10 | MRR |
|---|---|---|---|
| abstention | 30 | n/a | n/a |
| knowledge-update | 72 | 0.912 [0.850, 0.961] | 0.585 [0.508, 0.662] |
| multi-session | 121 | 0.770 [0.718, 0.819] | 0.651 [0.589, 0.713] |
| single-session-assistant | 56 | 0.982 [0.946, 1.000] | 0.689 [0.597, 0.773] |
| single-session-preference | 30 | 0.844 [0.722, 0.944] | 0.595 [0.464, 0.735] |
| single-session-user | 64 | 0.883 [0.805, 0.953] | 0.627 [0.536, 0.719] |
| temporal-reasoning | 127 | 0.745 [0.681, 0.807] | 0.571 [0.503, 0.642] |
| ALL | 500 | 0.830 [0.802, 0.859] | 0.617 [0.585, 0.653] |


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
