import asyncio
import json
from types import SimpleNamespace

from agentmem import agent_anthropic, agent_openai
from agentmem.agents_common import McpTools, grade
from agentmem.loaders import load_domain
from agentmem.mcp_eval import parse, server_params
from mcp import ClientSession
from mcp.client.stdio import stdio_client


def test_grade():
    assert grade("Production database.", "production database", "knowledge-update")
    assert grade("It took 12 days", "12", "temporal") and not grade("112 days", "12", "temporal")
    assert grade("confirmed data exfiltration; legal notified", "confirmed data exfiltration; legal notified", "single-hop")
    assert not grade("confirmed data exfiltration", "confirmed data exfiltration; legal notified", "single-hop")
    # phrasing differences must not count as misses...
    assert grade("T-4003 closed after confirming data exfiltration; legal was notified, host isolated.",
                 "confirmed data exfiltration; legal notified", "single-hop")
    assert grade("Partial data exfiltration confirmed, forensics ongoing.",
                 "data exfiltration confirmed, partial; forensics ongoing", "single-hop")
    assert grade("Confirmed exfiltration of data, customer data was not affected.",
                 "confirmed data exfiltration; no customer data affected", "single-hop")
    # ...but a different outcome must still fail
    assert not grade("Confirmed data exfiltration; no customer data affected.",
                     "confirmed data exfiltration; legal notified", "single-hop")
    assert not grade("Confirmed data exfiltration; legal notified.",
                     "confirmed data exfiltration; no customer data affected", "single-hop")
    assert grade("No record of that.", "no record", "abstention") and not grade("Yes, 10.0.0.1", "no record", "abstention")


async def _alerts():
    async with stdio_client(server_params("agentmem.mcp_alerts")) as (r, w), ClientSession(r, w) as s:
        await s.initialize()
        assert {t.name for t in (await s.list_tools()).tools} == {
            "search_alerts", "get_host", "get_user", "open_ticket_history"}
        old = next(q for q in load_domain("datasets/securitydomain") if "outcome of ticket" in q.text)
        ticket = old.text.split("ticket ")[1].rstrip("?")
        res = parse(await s.call_tool("open_ticket_history", {"ticket_id": ticket}))
        assert res["status"] in ("live", "archived")
        recent = parse(await s.call_tool("search_alerts", {"query": "login", "limit": 50}))
        assert all(x["timestamp"] >= "2026-10-28" for x in recent)  # live window only


def test_alert_server_over_mcp():
    asyncio.run(_alerts())


class Stub:
    """Scripted stand-in for the model: call recall once, then answer with the top record."""

    def __init__(self):
        self.messages, self.n = self, 0

    async def create(self, **kw):
        self.n += 1
        if self.n == 1:
            q = kw["messages"][0]["content"]
            use = SimpleNamespace(type="tool_use", id="t1", name="recall", input={"query": q, "k": 3})
            return SimpleNamespace(stop_reason="tool_use", content=[use])
        top = json.loads(kw["messages"][-1]["content"][0]["content"])[0]["text"]
        return SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text", text=top)])


async def _agent_loop(rerank):
    q = next(q for q in load_domain("datasets/securitydomain") if q.qtype == "single-hop")
    async with McpTools(rerank) as tools:
        assert {"recall", "store_memory", "rerank", "search_alerts"} <= set(tools.sessions)
        assert json.loads(await tools.call("get_host", {})).get("error")  # bad args surface as errors
        text, calls = await agent_anthropic.answer(q.text, tools, client=Stub())
    assert [c["name"] for c in calls] == ["recall"]
    assert grade(text, q.answer, q.qtype), (text, q.answer)


def test_anthropic_agent_loop_against_real_mcp_servers_with_stub_model():
    asyncio.run(_agent_loop(rerank=True))
    asyncio.run(_agent_loop(rerank=False))


def test_memory_service_is_thread_safe():
    """The MCP server runs tool calls on worker threads; concurrent recalls must not break."""
    from concurrent.futures import ThreadPoolExecutor

    from agentmem.domain import ASKED_AT
    from agentmem.service import MemoryService

    qs = load_domain("datasets/securitydomain")
    svc = MemoryService(now=ASKED_AT, use_ce=False)
    svc.bulk_load(qs[0].memories[:400])
    texts = [q.text for q in qs[:12]]
    with ThreadPoolExecutor(4) as pool:
        out = list(pool.map(lambda t: svc.recall(t, k=3, rerank=len(t) % 2 == 0), texts))
    assert all(len(o) == 3 for o in out)


def test_openai_agent_builds():
    agent, memory, alerts = agent_openai.build(rerank=True)
    assert agent.mcp_servers == [memory, alerts]


if __name__ == "__main__":
    test_grade()
    test_memory_service_is_thread_safe()
    test_openai_agent_builds()
    test_alert_server_over_mcp()
    test_anthropic_agent_loop_against_real_mcp_servers_with_stub_model()
    print("stage4 ok")
