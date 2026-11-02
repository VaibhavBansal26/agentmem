from dataclasses import dataclass, field

SOURCES = ("user_message", "assistant_message", "tool_call", "agent_note")


@dataclass(frozen=True)
class Memory:
    id: str
    text: str
    timestamp: float
    source: str = "user_message"
    entities: tuple = ()
    importance: float = 0.0


@dataclass
class Question:
    id: str
    text: str
    qtype: str
    answer: str
    gold_ids: tuple
    asked_at: float
    memories: list = field(default_factory=list)
