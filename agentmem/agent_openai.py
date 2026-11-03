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
    memory = stdio("agentmem.mcp_memory", AGENTMEM_RERANK="on" if rerank else "off",
                   AGENTMEM_SEED_DIR="datasets/securitydomain")
    alerts = stdio("agentmem.mcp_alerts")
    agent = Agent(name="investigator", instructions=SYSTEM_PROMPT, model=model,
                  mcp_servers=[memory, alerts])
    return agent, memory, alerts


def tool_calls(result):
    out = []
    for item in result.new_items:
        if getattr(item, "type", "") == "tool_call_item":
            raw = item.raw_item
            out.append({"name": getattr(raw, "name", "?"), "args": getattr(raw, "arguments", "")})
    return out


async def run(questions, rerank):
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY in your shell to run the OpenAI agent.")
    agent, memory, alerts = build(rerank)
    async with memory, alerts:
        for q in questions:
            res = await Runner.run(agent, q.text, max_turns=10)
            yield q, str(res.final_output), tool_calls(res)
