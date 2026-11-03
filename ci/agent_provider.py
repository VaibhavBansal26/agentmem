"""promptfoo provider: runs the Anthropic-SDK agent against the real MCP servers (reranker on).

The MCP servers are started once and reused for every test case (promptfoo keeps this worker alive).
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from memorank import agent_anthropic  # noqa: E402
from memorank.agents_common import McpTools  # noqa: E402
from memorank.run_agents import load_env  # noqa: E402

load_env()
_loop, _tools = None, None


def call_api(prompt, options, context):
    global _loop, _tools
    if _loop is None:
        _loop = asyncio.new_event_loop()
        _tools = McpTools(rerank=True)
        _loop.run_until_complete(_tools.__aenter__())
    text, calls = _loop.run_until_complete(agent_anthropic.answer(prompt, _tools))
    return {"output": text, "metadata": {"tool_calls": len(calls)}}
