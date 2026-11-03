# Pre-registration: reranker A/B (written before any non-smoke session was run)

**Question.** Does the full reranker pipeline (variant B) raise answer accuracy over dense top-k (variant A)
for an agent that queries memory through MCP?

**Setup.** OpenAI Agents SDK agent, `gpt-4o-mini`, same system prompt, tools and servers in both arms. Variant is
assigned per session by the flag service (`agentmem/flags.py`: SHA-256 hash of flag name + session id, 50/50, sticky).
A session = one question from the 85-question synthetic security set, drawn with replacement (seed 0), asked once.
Grading is the deterministic grader in `agentmem/agents_common.py`, unchanged from the Stage 4 runs.

**Sample size.** 340 sessions. A 50/50 hash split of exactly 300 would leave one arm below 150 about half the time.
Minimum 150 sessions per arm. If an arm ends below 150, add sessions (a size rule only, never a p-value rule) until
both reach 150. The Stage 4 run saw +0.15 for this agent (0.62 to 0.78); 150 per arm gives about 80% power for a
difference of that size. A smaller true effect would be underpowered, and a non-promotion would then not show "no effect".

**Primary analysis (one look, after the run is complete).** Two-sided pooled two-proportion z-test on accuracy,
alpha = 0.05. **Promote B only if** every condition holds: both arms have at least 150 sessions; p < 0.05 with B > A;
and the question-clustered bootstrap 95% CI for (B - A) excludes 0. The clustered CI is there because the 85 questions
repeat across sessions, so sessions are not independent. No other test counts toward the decision.

**Handling of problems.** Sessions with tool errors stay in the primary analysis (intention to treat); a sensitivity row
excludes them. The runner aborts after 3 consecutive error sessions; the server is fixed and the run resumed, never
re-drawn. No sessions are dropped for any other reason.

**Descriptive only (no decision weight).** Per-question-type accuracy, tool calls, tokens and latency per session.

**Known limits.** Synthetic, in-distribution data; one model; a small repeated question pool; reranker weights were
frozen from earlier stages and are not retuned here.
