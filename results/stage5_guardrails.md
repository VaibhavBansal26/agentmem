# Stage 5: guardrails

## Topical rail (refuse to speculate about a person's intent, motive or guilt): rule-based

`agentmem/guardrails.py::TopicalRail`, keyword and pattern rules, no model.

| set | what | result |
|---|---|---|
| 85 domain questions (benign) | the Stage 3 question set | 0 wrongly blocked |
| 12 handwritten intent prompts | written by me while building the rules (a development set) | all 12 blocked after fixing two rule gaps found on it |
| **60 held-out prompts** (`ci/guardrail_heldout.jsonl`) | 30 speculation + 30 benign, written by a separate model that never saw the rules; scored once, rules not changed afterwards | **recall 16/30 = 0.53; false positives 3/30 = 0.10; precision of blocks 16/19 = 0.84** |

Misses are mostly paraphrase and indirect phrasing ("what's the angle with that admin", "is the new hire trustworthy",
"did she know she was breaking policy"). False positives are "malicious" used about files and traffic, and "intended
scope" of a scan. **Conclusion: this is a cheap first-pass filter, not a dependable guardrail.** Keyword rules cannot
cover paraphrase. The next step would be an LLM intent classifier behind this filter, evaluated on a fresh held-out
set (this one is now used up for tuning purposes).

## Llama Guard 3 (safety classifier): implemented, not run against a real model

`LlamaGuard` calls a local Ollama server (`ollama pull llama-guard3:1b`) and parses its `safe` / `unsafe\nS1,...` reply.
It was not installed when this was built, so only the request shape, parsing and fail-closed behaviour are tested, against
a fake server (`tests/test_guardrails.py`). Note that Llama Guard's categories are general hazards (violence, crime, and
so on); it does not cover "don't speculate about an employee", which is why the topical rail exists separately.
Both guards fail closed: an unreachable classifier blocks.

## NeMo Guardrails

`nemo_rails/` is a NeMo Guardrails config (Colang flow `check topic` plus a custom action) that runs the topical rail as
an input rail with **no LLM configured**. Checked in a separate virtualenv (nemoguardrails 0.24.1): a speculation
question returns the refusal; an allowed question passes through unchanged. NeMo supplies the rail wiring only. The
classifier inside is still the rule-based `TopicalRail`, so its held-out numbers above are unchanged. NeMo is not in
`requirements-ci.txt` because its dependency tree is large and may conflict with the pinned SDKs; install it into its own
environment (`pip install nemoguardrails`) and run `tests/test_nemo.py`, which skips itself if NeMo is missing.
