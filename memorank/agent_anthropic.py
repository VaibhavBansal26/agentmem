"""Thin agent on the Anthropic SDK (manual tool-use loop) talking to both MCP servers."""
import os

from .agents_common import SYSTEM_PROMPT, McpTools

MODEL = "claude-haiku-5-5"


async def answer(question, tools, client=None, model=MODEL, max_turns=8):
    """Run one question. Returns (answer_text, calls) where calls is [{name, args}, ...]."""
    if client is None:
        import anthropic

        client = anthropic.AsyncAnthropic()
    messages, calls = [{"role": "user", "content": question}], []
    for _ in range(max_turns):
        resp = await client.messages.create(model=model, max_tokens=500, system=SYSTEM_PROMPT,
                                            tools=tools.tools, messages=messages)
        uses = [b for b in resp.content if b.type == "tool_use"]
        if resp.stop_reason != "tool_use" or not uses:
            return "".join(b.text for b in resp.content if b.type == "text").strip(), calls
        messages.append({"role": "assistant", "content": resp.content})
        results = []
        for u in uses:
            out = await tools.call(u.name, u.input)
            calls.append({"name": u.name, "args": u.input, "error": out.startswith('{"error"')})
            results.append({"type": "tool_result", "tool_use_id": u.id, "content": out})
        messages.append({"role": "user", "content": results})
    return "", calls


async def run(questions, rerank):
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("Set ANTHROPIC_API_KEY in your shell to run the Anthropic agent.")
    async with McpTools(rerank) as tools:
        for q in questions:
            yield q, *(await answer(q.text, tools))
