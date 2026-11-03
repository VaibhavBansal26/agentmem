# Stage 2: LLM time-scope parse vs regex (k=10, retrieval only)

Labels come from a Haiku subagent that saw only each question and its date (no haystack, no answer). Same pool, split and protocol as `stage2_ce.md`.

Tuned with CE, regex scope only: `{'dense': 1.0, 'ce': 2, 'scope_llm': 0.0, 'bm25': 2, 'scope': 4, 'entity': 0.0, 'recency': 0.5, 'recency_cue': 0.5}`

Tuned with CE + LLM scope: `{'dense': 1.0, 'ce': 0.5, 'scope_llm': 4, 'bm25': 2, 'scope': 0.0, 'entity': 0.0, 'recency': 0.25, 'recency_cue': 0.0}` (fit objective 0.798)


## LongMemEval_s dev half

| config | recall@10 | MRR |
|---|---|---|
| CE only | 0.895 [0.860, 0.928] | 0.693 [0.647, 0.739] |
| tuned, CE + regex scope | 0.871 [0.833, 0.908] | 0.722 [0.676, 0.767] |
| tuned, CE + LLM scope | 0.850 [0.811, 0.890] | 0.726 [0.676, 0.771] |
| CE + LLM scope, minus scope_llm | 0.846 [0.804, 0.886] | 0.723 [0.673, 0.768] |

Only the 32 questions with an LLM window:

| config | recall@10 | MRR |
|---|---|---|
| CE only | 0.688 [0.532, 0.831] | 0.493 [0.349, 0.635] |
| tuned, CE + regex scope | 0.742 [0.597, 0.887] | 0.480 [0.340, 0.613] |
| tuned, CE + LLM scope | 0.661 [0.500, 0.806] | 0.490 [0.342, 0.641] |
| CE + LLM scope, minus scope_llm | 0.629 [0.468, 0.790] | 0.466 [0.316, 0.617] |

## LoCoMo (held out, every 4th q)

| config | recall@10 | MRR |
|---|---|---|
| CE only | 0.728 [0.692, 0.763] | 0.572 [0.536, 0.610] |
| tuned, CE + regex scope | 0.711 [0.672, 0.746] | 0.583 [0.545, 0.619] |
| tuned, CE + LLM scope | 0.672 [0.634, 0.708] | 0.536 [0.499, 0.573] |
| CE + LLM scope, minus scope_llm | 0.661 [0.622, 0.698] | 0.527 [0.489, 0.566] |

Only the 61 questions with an LLM window:

| config | recall@10 | MRR |
|---|---|---|
| CE only | 0.686 [0.572, 0.803] | 0.505 [0.393, 0.611] |
| tuned, CE + regex scope | 0.656 [0.544, 0.775] | 0.494 [0.378, 0.601] |
| tuned, CE + LLM scope | 0.703 [0.592, 0.817] | 0.532 [0.427, 0.646] |
| CE + LLM scope, minus scope_llm | 0.614 [0.494, 0.739] | 0.463 [0.348, 0.582] |

### LongMemEval_s (all 500): scope coverage (questions with a window)

| type | n | regex | LLM | both |
|---|---|---|---|---|
| abstention | 30 | 1 | 2 | 1 |
| knowledge-update | 72 | 0 | 1 | 0 |
| multi-session | 121 | 10 | 27 | 9 |
| single-session-assistant | 56 | 0 | 0 | 0 |
| single-session-preference | 30 | 4 | 7 | 4 |
| single-session-user | 64 | 4 | 4 | 4 |
| temporal-reasoning | 127 | 20 | 28 | 19 |
| ALL | 500 | 39 | 69 | 37 |

Of 67 questions with an LLM window and gold evidence in the pool, 39 (58%) have at least one gold turn inside the window.

### LoCoMo (497): scope coverage (questions with a window)

| type | n | regex | LLM | both |
|---|---|---|---|---|
| adversarial | 113 | 6 | 10 | 5 |
| multi-hop | 76 | 1 | 4 | 1 |
| open-domain | 20 | 0 | 1 | 0 |
| single-hop | 210 | 16 | 42 | 15 |
| temporal | 78 | 1 | 4 | 1 |
| ALL | 497 | 24 | 61 | 22 |

Of 60 questions with an LLM window and gold evidence in the pool, 34 (57%) have at least one gold turn inside the window.
