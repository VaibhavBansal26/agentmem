"""Per-session outcome logging. Local SQLite is the source of truth for the A/B analysis.

LangfuseSink mirrors each session to Langfuse (variant as a tag) when LANGFUSE_* keys are set.
It is UNTESTED: no Langfuse account was available when it was written.
"""
import os
import sqlite3
import threading
import time

COLUMNS = ("session_id", "variant", "question_id", "qtype", "correct", "tool_calls", "tool_errors",
           "input_tokens", "output_tokens", "latency_s", "model", "answer", "ts")


class LocalSink:
    def __init__(self, db):
        self.db = sqlite3.connect(db, check_same_thread=False)
        self.lock = threading.Lock()
        self.db.execute(f"CREATE TABLE IF NOT EXISTS sessions ({', '.join(COLUMNS)}, PRIMARY KEY (session_id))")

    def done(self):
        with self.lock:
            return {r[0] for r in self.db.execute("SELECT session_id FROM sessions")}

    def record(self, **row):
        row.setdefault("ts", time.time())
        with self.lock:
            self.db.execute(f"INSERT OR REPLACE INTO sessions VALUES ({','.join('?' * len(COLUMNS))})",
                            [row[c] for c in COLUMNS])
            self.db.commit()

    def rows(self):
        with self.lock:
            cur = self.db.execute("SELECT * FROM sessions ORDER BY session_id")
            return [dict(zip(COLUMNS, r)) for r in cur]


class LangfuseSink:
    """Optional mirror. Construct only if LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are set."""

    def __init__(self):
        from langfuse import Langfuse

        self.client = Langfuse()

    def record(self, **row):
        trace = self.client.trace(name="memorank-session", id=row["session_id"], tags=[f"variant:{row['variant']}"],
                                  input=row["question_id"], output=row["answer"],
                                  metadata={k: row[k] for k in ("qtype", "tool_calls", "tool_errors", "latency_s")})
        trace.score(name="correct", value=float(row["correct"]))
        self.client.flush()


def sinks(db):
    out = [LocalSink(db)]
    if os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"):
        out.append(LangfuseSink())
    return out
