from .cache import cached

DEFAULT_MODEL = "claude-haiku-5-5"

PROMPT = """Answer the question using only the memories below. If they do not contain the answer, say "unknown".

Memories (oldest first):
{memories}

Current date: {now}
Question: {question}
Answer briefly:"""


def build_prompt(question: str, memories, now: str) -> str:
    ordered = sorted(memories, key=lambda m: m.timestamp)
    lines = "\n".join(f"- {m.text}" for m in ordered)
    return PROMPT.format(memories=lines, now=now, question=question)


def answer(prompt: str, model: str = DEFAULT_MODEL) -> str:
    """One fixed answerer for every experiment; cached by (model, prompt)."""

    def call():
        try:
            import anthropic
        except ImportError as e:
            raise RuntimeError("pip install anthropic to use --answerer") from e
        resp = anthropic.Anthropic().messages.create(
            model=model,
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text

    return cached("answers", [model, prompt], call)
