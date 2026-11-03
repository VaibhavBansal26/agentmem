"""Quickstart: python examples/quickstart.py   (pip install 'agentmem[dense]' for the evaluated pipeline)."""
from agentmem import Memory

mem = Memory("quickstart.db")  # a SQLite file; use ":memory:" for a throwaway store

mem.add("Host web-12 role set to staging web server", entities=["web-12"], timestamp="2026-10-02")
mem.add("Host web-12 role changed from staging web server to production web server", timestamp="2026-10-20")
mem.add("Antivirus scan completed on web-12 with no findings", timestamp="2026-10-25")
mem.add("Ticket T-4003 closed: confirmed data exfiltration; legal notified", entities=["T-4003"])

for question in ("What is the current role of web-12?", "What was the outcome of ticket T-4003?"):
    print(question)
    for hit in mem.recall(question, k=2):
        print(f"  {hit.rank}. [{hit.time:%Y-%m-%d}] {hit.text}")

# Bring your own models: any function works, nothing is downloaded.
#   Memory(embedder=lambda texts: my_embed(texts), scorer=lambda pairs: my_reranker(pairs))
# Expose it to an agent:
#   from agentmem.adapters import anthropic_tools, openai_agents_tools
