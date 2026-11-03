# Budget ledger (target: under $15 total)

Log every paid API call batch here.

| date | stage | what | model | tokens in/out | cost (USD) |
|---|---|---|---|---|---|
| 2026-11-02 | 2 | LLM time-scope labels (997 questions) | Haiku via subagent, no API key used | not metered | $0.00 API |

| 2026-11-02 | 4 | Agent evaluation: 85 domain questions x 2 agents x reranker off/on (plus 5-question smoke tests and one aborted, discarded OpenAI run) | claude-haiku-5-5; gpt-4o-mini | not metered (logs do not record tokens) | unknown; check the Anthropic and OpenAI dashboards and fill in |
| 2026-11-02 | 5 | Pre-registered A/B: 340 sessions, gpt-4o-mini via OpenAI Agents SDK (metered from SDK usage) | gpt-4o-mini | 646,137 in / 16,839 out | not computed; multiply by the current OpenAI price |

Running total: unknown for Stage 4 (agent runs unmetered). Everything else is $0.00. Add token logging before further paid runs.
