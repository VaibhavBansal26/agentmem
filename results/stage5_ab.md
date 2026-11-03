# Stage 5: online A/B, reranker A (dense top-k) vs B (full pipeline)

Sessions: A=153, B=187 (pre-registered minimum 150 per arm).

| arm | n | accuracy | tool calls | tool-error rows | input tok | output tok | latency s |
|---|---|---|---|---|---|---|---|
| A | 153 | 0.621 | 1.28 | 0 | 1936 | 49 | 2.2 |
| B | 187 | 0.797 | 1.27 | 0 | 1871 | 50 | 4.0 |

Difference B - A: +0.176. Two-proportion z = 3.58, two-sided p = 0.0003.
Question-clustered bootstrap 95% CI for the difference: [+0.065, +0.281].

Sensitivity, excluding sessions with tool errors: A 0.621 (n=153), B 0.797 (n=187).

**Decision under the pre-registered rule: PROMOTE B.**

| type | n(A) | acc A | n(B) | acc B |
|---|---|---|---|---|
| abstention | 15 | 1.00 | 26 | 1.00 |
| knowledge-update | 33 | 0.94 | 36 | 0.92 |
| multi-hop | 35 | 0.14 | 39 | 0.54 |
| single-hop | 35 | 0.89 | 46 | 1.00 |
| temporal | 35 | 0.37 | 40 | 0.57 |
