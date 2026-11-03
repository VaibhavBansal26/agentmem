"""MCP server for a live alert queue over the synthetic security history.

It only sees the last AGENTMEM_LIVE_DAYS days (default 2) of alerts, like a SIEM's hot tier.
Older activity is archived, so questions about earlier incidents need the memory server.
Environment: AGENTMEM_SEED_DIR (default datasets/securitydomain), AGENTMEM_NOW, AGENTMEM_LIVE_DAYS.
"""
import os
import re

from mcp.server.mcpserver import MCPServer

from .domain import ASKED_AT, DAY
from .loaders import load_domain
from .rerank import id_tokens
from .service import iso

mcp = MCPServer("agentmem-alerts", instructions="Live alert queue (recent events only).")
_state = {}


def state():
    if not _state:
        mems = load_domain(os.environ.get("AGENTMEM_SEED_DIR", "datasets/securitydomain"))[0].memories
        now = float(os.environ.get("AGENTMEM_NOW", ASKED_AT))
        days = float(os.environ.get("AGENTMEM_LIVE_DAYS", "2"))
        _state.update(all=mems, now=now,
                      live=[m for m in mems if now - days * DAY < m.timestamp <= now],
                      known=set().union(*(id_tokens(m.text) for m in mems)))
    return _state


def row(m):
    return {"id": m.id, "timestamp": iso(m.timestamp), "text": m.text}


@mcp.tool()
def search_alerts(query: str, host: str | None = None, user: str | None = None, limit: int = 10) -> list[dict]:
    """Search recent alerts (live window only) by keywords, optionally for one host or user."""
    words = set(re.findall(r"[a-z0-9][\w.:/-]*", query.lower())) - {"the", "a", "of", "for", "to", "on"}
    hits = []
    for m in state()["live"]:
        toks = id_tokens(m.text)
        if (host and host.lower() not in toks) or (user and user.lower() not in toks):
            continue
        score = len(words & toks)
        if score or (not words and (host or user)):
            hits.append((score, m.timestamp, m))
    hits.sort(key=lambda h: (-h[0], -h[1]))
    return [row(m) for _, _, m in hits[:limit]]


def _entity(name, kind):
    s = state()
    key = name.lower()
    recent = [m for m in s["live"] if key in id_tokens(m.text)]
    return {kind: name, "known": key in s["known"], "alerts_in_live_window": len(recent),
            "latest_alert": row(recent[-1]) if recent else None,
            "note": "Role/department and older history are not in the live queue; ask memory."}


@mcp.tool()
def get_host(host: str) -> dict:
    """Inventory view of a host: whether it exists and its recent alert activity."""
    return _entity(host, "host")


@mcp.tool()
def get_user(user: str) -> dict:
    """Inventory view of a user account: whether it exists and its recent alert activity."""
    return _entity(user, "user")


@mcp.tool()
def open_ticket_history(ticket_id: str) -> dict:
    """Records for a ticket id (e.g. T-4003) if it is still in the live window."""
    key = ticket_id.lower()
    recs = [row(m) for m in state()["live"] if key in id_tokens(m.text)]
    if recs:
        return {"ticket": ticket_id, "status": "live", "records": recs}
    return {"ticket": ticket_id, "status": "archived",
            "note": "Older than the live window; check agent memory."}


if __name__ == "__main__":
    mcp.run("stdio")
