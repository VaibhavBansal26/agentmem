"""MCP server exposing the agent memory: store_memory, recall, rerank.

Environment: AGENTMEM_DB (sqlite path, default in-memory), AGENTMEM_RERANK=on|off (variant B / A),
AGENTMEM_CE=1|0 (cross-encoder), AGENTMEM_NOW (epoch seconds, for replaying a fixed history),
AGENTMEM_SEED_DIR (dataset directory with memories.jsonl to preload).
"""
import os

from mcp.server.mcpserver import MCPServer

from .loaders import load_domain
from .service import MemoryService

mcp = MCPServer(
    "agentmem-memory",
    instructions="Long-term memory for a security-investigation agent. Use recall to look up past "
                 "alerts, tickets and notes; use store_memory to save decisions and confirmed facts.",
)
_svc = None


def svc():
    global _svc
    if _svc is None:
        now = os.environ.get("AGENTMEM_NOW")
        _svc = MemoryService(
            os.environ.get("AGENTMEM_DB", ":memory:"),
            rerank=os.environ.get("AGENTMEM_RERANK", "on") == "on",
            use_ce=os.environ.get("AGENTMEM_CE", "1") == "1",
            now=float(now) if now else None,
        )
        seed = os.environ.get("AGENTMEM_SEED_DIR")
        if seed and not _svc.memories:
            _svc.bulk_load(load_domain(seed)[0].memories)
    return _svc


@mcp.tool()
def store_memory(text: str, source: str = "agent_note", entities: list[str] | None = None,
                 importance: float = 0.0) -> str:
    """Save a memory (a decision, confirmed fact or note). Returns its id."""
    return svc().store(text, source, entities or (), importance)


@mcp.tool()
def recall(query: str, k: int = 8, start_date: str | None = None, end_date: str | None = None) -> list[dict]:
    """Find the memories most relevant to a question. Optional start_date/end_date (YYYY-MM-DD)
    restrict results to that window. Results are ranked best first."""
    return svc().recall(query, k, start_date, end_date)


@mcp.tool()
def rerank(query: str, candidates: list[dict]) -> list[dict]:
    """Rerank candidate records ({text, optional timestamp as ISO date, optional id}) for a query."""
    return svc().rerank_candidates(query, candidates)


def main():
    """Console entry point: `agentmem-mcp` (configure with the AGENTMEM_* environment variables)."""
    mcp.run("stdio")


if __name__ == "__main__":
    main()
