"""Send every domain question through the real MCP `recall` tool, reranker off (A) vs on (B).

No LLM involved: this measures what an agent would be handed, through the actual protocol.
"""
import asyncio
import json
import os
import sys
import time
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from . import metrics
from .domain import ASKED_AT
from .loaders import load_domain


def server_params(module, **env):
    base = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH",)}
    return StdioServerParameters(command=sys.executable, args=["-m", module],
                                 env={**base, "MEMORANK_NOW": str(ASKED_AT), **env})


def parse(res):
    """Tool result -> python object, whatever the SDK's result shape."""
    sc = getattr(res, "structured_content", None) or getattr(res, "structuredContent", None)
    if sc is not None:
        return sc.get("result", sc) if isinstance(sc, dict) else sc
    return json.loads(res.content[0].text)


async def run_variant(rerank, questions, k, data_dir):
    params = server_params("memorank.mcp_memory", MEMORANK_RERANK="on" if rerank else "off",
                           MEMORANK_SEED_DIR=data_dir)
    rows, lat = [], []
    async with stdio_client(params) as (r, w), ClientSession(r, w) as session:
        await session.initialize()
        tools = sorted(t.name for t in (await session.list_tools()).tools)
        for q in questions:
            t = time.time()
            res = await session.call_tool("recall", {"query": q.text, "k": k})
            lat.append(time.time() - t)
            ids = [x["id"] for x in parse(res)]
            rows.append((q.qtype, metrics.recall_at_k(ids, q.gold_ids, k), metrics.mrr(ids, q.gold_ids)))
    return rows, lat, tools


def fmt(vals):
    m, lo, hi, _ = metrics.mean_ci(vals)
    return "n/a" if m is None else f"{m:.3f} [{lo:.3f}, {hi:.3f}]"


async def main(k=10, data_dir="datasets/securitydomain"):
    questions = load_domain(data_dir)
    out = [f"# Stage 4: domain questions through the MCP `recall` tool (k={k}, no LLM)\n",
           f"{len(questions)} questions, one `recall` call each, over stdio to a separate server process. "
           "A = dense top-k (reranker off); B = Stage 2/3 reranker (cross-encoder, cue-gated "
           "entity-scoped recency, entity chaining) with weights frozen from earlier stages.\n"]
    results = {}
    for name, rr in (("A (off)", False), ("B (on)", True)):
        results[name] = await run_variant(rr, questions, k, data_dir)
    out.append(f"Tools exposed: `{results['A (off)'][2]}`\n")
    out.append("| variant | recall@10 | MRR | median recall() latency |\n|---|---|---|---|")
    for name, (rows, lat, _) in results.items():
        lat = sorted(lat)
        out.append(f"| {name} | {fmt([r[1] for r in rows])} | {fmt([r[2] for r in rows])} | "
                   f"{lat[len(lat) // 2] * 1000:.0f} ms |")
    types = sorted({q.qtype for q in questions})
    out.append("\nPer type (recall@10 / MRR):\n\n| type | n | A (off) | B (on) |\n|---|---|---|---|")
    for t in types:
        cells = []
        for name in results:
            sel = [r for r in results[name][0] if r[0] == t]
            cells.append(f"{fmt([r[1] for r in sel])} / {fmt([r[2] for r in sel])}")
        out.append(f"| {t} | {sum(q.qtype == t for q in questions)} | " + " | ".join(cells) + " |")
    text = "\n".join(out) + "\n"
    print(text)
    Path("results").mkdir(exist_ok=True)
    Path("results/stage4_mcp_recall.md").write_text(text)


if __name__ == "__main__":
    asyncio.run(main())
