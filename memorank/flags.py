"""Tiny feature-flag service: sticky, deterministic, weighted variant assignment in SQLite."""
import hashlib
import json
import sqlite3
import threading
import time


class Flags:
    def __init__(self, db=":memory:"):
        self.db = sqlite3.connect(db, check_same_thread=False)
        self.lock = threading.Lock()
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS flags (name TEXT PRIMARY KEY, weights TEXT);
            CREATE TABLE IF NOT EXISTS assignments (
                session_id TEXT, flag TEXT, variant TEXT, ts REAL, PRIMARY KEY (session_id, flag));""")

    def set(self, name, weights):
        """weights: {"A": 0.5, "B": 0.5}. Existing assignments are kept (sticky)."""
        with self.lock:
            self.db.execute("INSERT OR REPLACE INTO flags VALUES (?, ?)", (name, json.dumps(weights)))
            self.db.commit()

    def variant(self, name, session_id):
        with self.lock:
            row = self.db.execute("SELECT variant FROM assignments WHERE session_id=? AND flag=?",
                                  (session_id, name)).fetchone()
            if row:
                return row[0]
            weights = json.loads(self.db.execute("SELECT weights FROM flags WHERE name=?", (name,)).fetchone()[0])
            u = int(hashlib.sha256(f"{name}:{session_id}".encode()).hexdigest()[:8], 16) / 16 ** 8
            total, acc = sum(weights.values()), 0.0
            for variant, w in weights.items():
                acc += w / total
                if u < acc:
                    break
            self.db.execute("INSERT INTO assignments VALUES (?, ?, ?, ?)", (session_id, name, variant, time.time()))
            self.db.commit()
            return variant
