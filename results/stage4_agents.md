# Stage 4: agents through MCP, reranker off vs on

| run | n | accuracy | mean tool calls | mean recall() calls | rows with tool errors |
|---|---|---|---|---|---|
| anthropic_baseline | 85 | 0.894 | 3.06 | 1.92 | 0 |
| anthropic_reranker | 85 | 0.894 | 2.69 | 1.69 | 0 |
| openai_baseline | 85 | 0.624 | 1.29 | 1.14 | 0 |
| openai_reranker | 85 | 0.776 | 1.25 | 1.11 | 0 |

## Paired comparison, reranker off (A) vs on (B)

Each question is asked in both variants; the difference is bootstrapped over questions (2000 resamples, 95% CI). Single run per cell, so model nondeterminism is not averaged out.

### anthropic

Accuracy 0.894 -> 0.894; paired difference +0.000 [-0.047, +0.047]; fixed by reranker 2, broken by reranker 2.

| type | n | off | on |
|---|---|---|---|
| abstention | 8 | 1.00 | 1.00 |
| knowledge-update | 18 | 1.00 | 1.00 |
| multi-hop | 20 | 0.90 | 0.90 |
| single-hop | 18 | 1.00 | 0.94 |
| temporal | 21 | 0.67 | 0.71 |

### openai

Accuracy 0.624 -> 0.776; paired difference +0.153 [+0.071, +0.247]; fixed by reranker 14, broken by reranker 1.

| type | n | off | on |
|---|---|---|---|
| abstention | 8 | 1.00 | 1.00 |
| knowledge-update | 18 | 0.89 | 0.94 |
| multi-hop | 20 | 0.20 | 0.55 |
| single-hop | 18 | 0.83 | 1.00 |
| temporal | 21 | 0.48 | 0.57 |

## Remaining misses by type

- anthropic_baseline: {'multi-hop': 2, 'temporal': 7}
- anthropic_reranker: {'single-hop': 1, 'multi-hop': 2, 'temporal': 6}
- openai_baseline: {'single-hop': 3, 'multi-hop': 16, 'temporal': 11, 'knowledge-update': 2}
- openai_reranker: {'multi-hop': 9, 'temporal': 9, 'knowledge-update': 1}
