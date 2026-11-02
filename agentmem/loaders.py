"""Dataset loaders -> list[Question]. One Memory per conversation turn.

Written from the published formats; verify against the real files on first use.
"""
import json
import re
from datetime import datetime, timezone

from .schema import Memory, Question


def _epoch(dt):
    return dt.replace(tzinfo=timezone.utc).timestamp()


def _lme_date(s):
    # "2023/05/20 (Sat) 02:21"
    m = re.match(r"(\d{4})/(\d{2})/(\d{2}).*?(\d{1,2}):(\d{2})", s)
    y, mo, d, h, mi = map(int, m.groups())
    return _epoch(datetime(y, mo, d, h, mi))


def _end_of_day(ts):
    return ts - ts % 86400 + 86400 - 1


def load_longmemeval(path, limit=0):
    data = json.load(open(path))
    out = []
    for item in data[: limit or None]:
        memories, gold = [], []
        sessions = zip(
            item["haystack_session_ids"], item["haystack_dates"], item["haystack_sessions"]
        )
        for sid, date, turns in sessions:
            base = _lme_date(date)
            for i, turn in enumerate(turns):
                mid = f"{sid}:{i}"
                source = "user_message" if turn["role"] == "user" else "assistant_message"
                memories.append(Memory(mid, turn["content"], base + i, source))
                if turn.get("has_answer"):
                    gold.append(mid)
        qtype = item["question_type"]
        if item["question_id"].endswith("_abs"):
            qtype, gold = "abstention", []
        out.append(
            Question(
                item["question_id"],
                item["question"],
                qtype,
                str(item["answer"]),
                tuple(gold),
                # Some sessions are logged later on the question's own day, so
                # treat the question date as the whole day, not its clock time.
                _end_of_day(_lme_date(item["question_date"])),
                memories,
            )
        )
    return out


LOCOMO_CATEGORIES = {
    1: "multi-hop",
    2: "temporal",
    3: "open-domain",
    4: "single-hop",
    5: "adversarial",
}


def _locomo_date(s):
    # "1:56 pm on 8 May, 2023"
    return _epoch(datetime.strptime(s.strip(), "%I:%M %p on %d %B, %Y"))


def load_locomo(path, limit=0):
    data = json.load(open(path))
    out = []
    for conv in data:
        c = conv["conversation"]
        memories = []
        n = 1
        while f"session_{n}" in c:
            base = _locomo_date(c[f"session_{n}_date_time"])
            for i, t in enumerate(c[f"session_{n}"]):
                memories.append(
                    Memory(t["dia_id"], f'{t["speaker"]}: {t["text"]}', base + i, "user_message")
                )
            n += 1
        asked_at = max(m.timestamp for m in memories) + 1
        for j, qa in enumerate(conv["qa"]):
            gold = tuple(e for ev in qa.get("evidence", []) for e in re.findall(r"D\d+:\d+", ev))
            out.append(
                Question(
                    f'{conv.get("sample_id", "conv")}-{j}',
                    qa["question"],
                    LOCOMO_CATEGORIES.get(qa["category"], str(qa["category"])),
                    str(qa.get("answer", qa.get("adversarial_answer", ""))),
                    gold,
                    asked_at,
                    memories,
                )
            )
    return out[: limit or None]


def load_toy(limit=0):
    """Tiny in-memory set covering the question types the reranker targets."""
    day = 86400.0
    t0 = _epoch(datetime(2026, 1, 1))

    def mem(i, d, text, source="agent_note"):
        return Memory(f"m{i}", text, t0 + d * day, source)

    noise = [
        mem(100 + i, i, f"Routine alert {i}: scheduled scan finished on host-{i} with no findings")
        for i in range(20)
    ]
    mems = noise + [
        mem(1, 2, "host-db1 role is staging database"),
        mem(2, 15, "host-db1 role changed to production database after migration"),
        mem(3, 3, "User alice failed login 5 times from 10.0.0.9"),
        mem(4, 17, "Ticket 4412 closed: alice exfiltrated data to 203.0.113.7"),
    ]
    now = t0 + 20 * day
    return [
        Question("toy-update", "What is the role of host-db1?", "knowledge-update",
                 "production database", ("m2",), now, mems),
        Question("toy-lookup", "What happened with alice login failures?", "single-hop",
                 "5 failed logins from 10.0.0.9", ("m3",), now, mems),
        Question("toy-exfil", "Which IP did alice send data to?", "multi-hop",
                 "203.0.113.7", ("m4",), now, mems),
    ][: limit or None]
