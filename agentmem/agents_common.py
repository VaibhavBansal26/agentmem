"""Shared pieces for the two thin agents: prompt, MCP connections, grading, logging."""
import asyncio
import json
import re
from contextlib import AsyncExitStack
from pathlib import Path

from mcp import ClientSession
from mcp.client.stdio import stdio_client

from .mcp_eval import parse, server_params

SYSTEM_PROMPT = """You are a security-investigation assistant. Answer the analyst's question about past \
activity using your tools: `recall` searches long-term memory of past alerts, tickets and notes; the \
alert tools show only the live queue (recent events). Older history is only in memory. Ground every \
answer in retrieved records; if the records do not contain the answer, say "no record". When facts \
changed over time, give the most recent value. Reply with just the answer, in one short sentence."""

ABSTAIN = ("no record", "unknown", "not found", "no evidence", "no data", "cannot", "could not find",
           "did not find", "don't have", "do not have", "no matching")


def norm(s):
    return re.sub(r"[^a-z0-9.\- ]+", " ", s.lower()).strip()


STOP = {"the", "a", "an", "was", "were", "is", "are", "be", "been", "being", "of", "to", "and", "that",
        "it", "its", "with", "by", "for", "in", "on", "at", "as", "had", "has", "have"}


def stem(w):
    if w == "not":
        return "no"
    if any(c.isdigit() for c in w):  # ids, IPs, numbers stay exact
        return w
    for suf in ("ing", "ed", "es", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[: -len(suf)]
    return w


def words(s):
    toks = (w.strip(".-") for w in norm(s).split())
    return {stem(w) for w in toks if w and w not in STOP}


def grade(pred, gold, qtype):
    """Deterministic grading. Abstention: the answer must decline. Numbers must match exactly.
    Otherwise every ';'- or ','-separated clause of the gold answer must have all of its (stemmed,
    stopword-free) words present in the answer, so 'confirming data exfiltration; legal was notified'
    matches 'confirmed data exfiltration; legal notified'."""
    p = norm(pred)
    if qtype == "abstention":
        return any(a in p for a in ABSTAIN)
    if gold.strip().isdigit():
        return re.search(rf"(?<![\d.]){re.escape(gold.strip())}(?![\d])", p) is not None
    have = words(pred)
    return all(words(c) <= have for c in re.split(r"[;,]", gold) if words(c))


class McpTools:
    """Spawns the memory and alert servers over stdio and routes tool calls by name."""

    def __init__(self, rerank):
        self.rerank, self.sessions, self.tools, self.stack = rerank, {}, [], AsyncExitStack()

    async def __aenter__(self):
        try:
            return await self._connect()
        except BaseException:
            try:  # don't let a half-open transport turn an error into a hang
                await asyncio.wait_for(self.stack.aclose(), 10)
            except BaseException:
                pass
            raise

    async def _connect(self):
        specs = [("agentmem.mcp_memory", {"AGENTMEM_RERANK": "on" if self.rerank else "off",
                                          "AGENTMEM_SEED_DIR": "datasets/securitydomain"}),
                 ("agentmem.mcp_alerts", {})]
        for module, env in specs:
            r, w = await self.stack.enter_async_context(stdio_client(server_params(module, **env)))
            session = await self.stack.enter_async_context(ClientSession(r, w))
            await session.initialize()
            for t in (await session.list_tools()).tools:
                self.sessions[t.name] = session
                schema = getattr(t, "input_schema", None) or getattr(t, "inputSchema", None)
                self.tools.append({"name": t.name, "description": t.description or "", "input_schema": schema})
        return self

    async def __aexit__(self, *exc):
        await self.stack.aclose()

    async def call(self, name, args):
        res = await self.sessions[name].call_tool(name, args)
        if getattr(res, "is_error", False) or getattr(res, "isError", False):
            text = res.content[0].text if res.content else "tool error"
            return json.dumps({"error": text})
        return json.dumps(parse(res))


def log_row(path, row):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


def summarize(paths, out="results/stage4_agents.md"):
    """Aggregate JSONL run logs (one file per agent x variant) into an accuracy/tool-call table."""
    lines = ["# Stage 4: agents through MCP, reranker off vs on\n",
             "| run | n | accuracy | mean tool calls | mean recall() calls | rows with tool errors |",
             "|---|---|---|---|---|---|"]
    for p in paths:
        rows = [json.loads(l) for l in open(p)]
        if not rows:
            continue
        acc = sum(grade(r["answer"], r["gold"], r["qtype"]) for r in rows) / len(rows)
        calls = sum(len(r["calls"]) for r in rows) / len(rows)
        rec = sum(sum(c["name"] == "recall" for c in r["calls"]) for r in rows) / len(rows)
        errs = sum(bool(r.get("tool_errors")) for r in rows)
        lines.append(f"| {Path(p).stem} | {len(rows)} | {acc:.3f} | {calls:.2f} | {rec:.2f} | {errs} |")
    Path(out).write_text("\n".join(lines) + "\n")
    return "\n".join(lines)
