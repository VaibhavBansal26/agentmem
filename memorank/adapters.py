"""Expose a Memory to agent frameworks as two tools: memory_recall and memory_store.

    tools, handle = anthropic_tools(mem)      # pass `tools` to messages.create; route tool_use blocks to handle()
    agent = Agent(..., tools=openai_agents_tools(mem))
"""
import json

RECALL_DOC = ("Search long-term memory for records relevant to a question. Results are best-first. "
              "Optional start_date/end_date (YYYY-MM-DD) restrict to a time window.")
STORE_DOC = "Save a decision, confirmed fact or note to long-term memory. Returns its id."


def _recall(mem, query, k=8, start_date=None, end_date=None):
    hits = mem.recall(query, k=k, start=start_date, end=end_date)
    return json.dumps([{"rank": h.rank, "id": h.id, "text": h.text,
                        "timestamp": h.time.strftime("%Y-%m-%dT%H:%M:%SZ"), "source": h.source} for h in hits])


def _store(mem, text, source="agent_note", entities=None):
    return json.dumps({"id": mem.add(text, source=source, entities=entities or ())})


def anthropic_tools(mem):
    """Returns (tool_definitions, handle). handle(name, input_dict) -> str result for a tool_result block."""
    tools = [
        {"name": "memory_recall", "description": RECALL_DOC, "input_schema": {
            "type": "object", "required": ["query"], "properties": {
                "query": {"type": "string"}, "k": {"type": "integer", "default": 8},
                "start_date": {"type": "string"}, "end_date": {"type": "string"}}}},
        {"name": "memory_store", "description": STORE_DOC, "input_schema": {
            "type": "object", "required": ["text"], "properties": {
                "text": {"type": "string"}, "source": {"type": "string", "default": "agent_note"},
                "entities": {"type": "array", "items": {"type": "string"}}}}},
    ]

    def handle(name, args):
        if name == "memory_recall":
            return _recall(mem, **args)
        if name == "memory_store":
            return _store(mem, **args)
        raise KeyError(f"unknown tool {name!r}")

    return tools, handle


def openai_agents_tools(mem):
    """Returns [memory_recall, memory_store] as OpenAI Agents SDK function tools (needs memorank[agents])."""
    from agents import function_tool

    @function_tool(strict_mode=False, name_override="memory_recall", description_override=RECALL_DOC)
    def memory_recall(query: str, k: int = 8, start_date: str | None = None, end_date: str | None = None) -> str:
        return _recall(mem, query, k, start_date, end_date)

    @function_tool(strict_mode=False, name_override="memory_store", description_override=STORE_DOC)
    def memory_store(text: str, source: str = "agent_note", entities: list[str] | None = None) -> str:
        return _store(mem, text, source, entities)

    return [memory_recall, memory_store]
