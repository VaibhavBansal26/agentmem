"""Thin agent on the OpenAI Agents SDK; the SDK connects to both MCP servers itself."""
import os

from agents import Agent, Runner
from agents.mcp import MCPServerStdio

from .agents_common import SYSTEM_PROMPT
from .mcp_eval import server_params

MODEL = "gpt-4o-mini"


def stdio(module, **env):
    p = server_params(module, **env)
    return MCPServerStdio(params={"command": p.command, "args": p.args, "env": p.env},
                          cache_tools_list=True, client_session_timeout_seconds=120)


def build(rerank, model=MODEL):
    memory = stdio("memorank.mcp_memory", MEMORANK_RERANK="on" if rerank else "off",
                   MEMORANK_SEED_DIR="datasets/securitydomain")
    alerts = stdio("memorank.mcp_alerts")
    agent = Agent(name="investigator", instructions=SYSTEM_PROMPT, model=model,
                  mcp_servers=[memory, alerts])
    return agent, memory, alerts


def tool_calls(result):
    """Tool calls in order, each flagged if its output looks like a tool error (the SDK hands errors
    to the model as plain text, so they would otherwise pass silently)."""
    out = []
    for item in result.new_items:
        kind = getattr(item, "type", "")
        if kind == "tool_call_item":
            raw = item.raw_item
            out.append({"name": getattr(raw, "name", "?"), "args": getattr(raw, "arguments", ""), "error": False})
        elif kind == "tool_call_output_item" and out:
            text = str(getattr(item, "output", ""))[:200].lower()
            out[-1]["error"] = text.startswith("error") or '"error"' in text[:30]
    return out


async def run(questions, rerank):
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY in your shell to run the OpenAI agent.")
    agent, memory, alerts = build(rerank)
    async with memory, alerts:
        for q in questions:
            res = await Runner.run(agent, q.text, max_turns=10)
            yield q, str(res.final_output), tool_calls(res)
