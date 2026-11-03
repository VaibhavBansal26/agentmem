"""Synthetic security-investigation history with planted incidents and exact ground truth.

`python -m memorank.domain` regenerates datasets/securitydomain/{memories,questions}.jsonl
from a fixed seed. Every question's gold records are known by construction.
"""
import json
import random
from datetime import datetime, timezone
from pathlib import Path

DAY = 86400.0
BASE = datetime(2026, 10, 1, tzinfo=timezone.utc).timestamp()
ASKED_AT = BASE + 29 * DAY
FIRST = ("alex bella chris dana eli fay gus hana ian jo kim leo mia noah olga pat quinn rae sam tia "
         "uma vik wes xin yara zed amir beth carl dina evan fran gabe hugo iris jack kara liam nina").split()
LAST = ("martinez chen okafor patel nguyen silva kowalski haddad ivanov jensen kapoor lopez morgan "
        "novak ortiz park reyes sato tanaka uribe vega wong yilmaz zhou adams brooks clark davis "
        "evans foster grant hayes irwin jones keller lane moore nash owens").split()
ROLES = ["staging web server", "production web server", "production database", "backup server",
         "file server", "jump host", "build agent", "reporting server"]
DEPTS = ["Finance", "IT", "Sales", "Legal", "Engineering", "HR", "Support"]
OUTCOMES = ["confirmed data exfiltration; legal notified",
            "confirmed data exfiltration; no customer data affected",
            "data exfiltration confirmed, partial; forensics ongoing"]


class Builder:
    def __init__(self, seed):
        self.rnd = random.Random(seed)
        self.recs = []
        self.ticket = 4000
        self.alert = 30000

    def add(self, day, text, source="agent_note", ents=(), importance=0.0):
        # Small jitter keeps the intended order of events within a day.
        r = {"ts": BASE + day * DAY + self.rnd.randint(0, 600),
             "text": text, "source": source, "entities": list(ents), "importance": importance}
        self.recs.append(r)
        return r

    def tid(self):
        self.ticket += self.rnd.randint(1, 9)
        return f"T-{self.ticket}"

    def aid(self):
        self.alert += self.rnd.randint(1, 40)
        return f"A-{self.alert}"


def build(seed=7):
    b = Builder(seed)
    rnd = b.rnd
    names = [f"{f[0]}{l}" for f in FIRST for l in LAST]
    rnd.shuffle(names)
    users = names[:200]
    kinds = ["web", "db", "app", "wks", "srv", "fs"]
    hosts = [f"{k}-{n:02d}" for k in kinds for n in range(1, 51)]
    rnd.shuffle(hosts)
    ext = [f"{p}.{i}" for p in ("198.51.100", "203.0.113", "192.0.2") for i in range(2, 250)]
    rnd.shuffle(ext)
    used_ext = set()

    def take_ext():
        ip = next(i for i in ext if i not in used_ext)
        used_ext.add(ip)
        return ip

    inc_users, rest_users = users[:12], users[12:]
    inc_hosts, other_hosts = hosts[:36], hosts[36:]
    incidents = []
    starts = list(range(1, 13))
    rnd.shuffle(starts)
    for i, U in enumerate(inc_users):
        a = starts[i]
        E, P1, P2 = inc_hosts[3 * i: 3 * i + 3]
        S, Z = take_ext(), take_ext()
        gb = round(rnd.uniform(0.8, 9.5), 1)
        n_fail = rnd.randint(3, 6)
        inc = {"U": U, "E": E, "P1": P1, "P2": P2, "S": S, "Z": Z, "gb": gb}
        inc["fails"] = [b.add(a + k * 0.02 + 0.1, f"[{b.aid()}] Failed login for user {U} on vpn-gw-01 from {S} "
                              f"(attempt {k + 1} of {n_fail})", "tool_call", [U, S, "vpn-gw-01"])
                        for k in range(n_fail)]
        inc["tkt"] = b.tid()
        inc["open"] = b.add(a + 0.9, f"[{inc['tkt']}] Ticket opened: suspected account compromise for user {U} "
                            f"after repeated failed logins from {S}", "agent_note", [U, S, inc["tkt"]], 0.6)
        inc["rtkt"] = b.tid()
        inc["reset"] = b.add(a + 1.2, f"[{inc['rtkt']}] Password reset completed for user {U} at analyst "
                             f"request after suspicious sign-in activity", "agent_note", [U, inc["rtkt"]], 0.7)
        inc["login"] = b.add(a + 1.6, f"[{b.aid()}] Successful login for user {U} on {E} from {S} outside "
                             f"normal hours", "tool_call", [U, E, S], 0.3)
        d1 = a + 2 + rnd.randint(0, 3)
        d2 = d1 + rnd.randint(1, 4)
        de = d2 + rnd.randint(2, 6)
        inc["lat1"] = b.add(d1 + 0.3, f"[{b.aid()}] User {U} authenticated to {P1} via RDP from {E}",
                            "tool_call", [U, P1, E], 0.3)
        inc["lat2"] = b.add(d2 + 0.3, f"[{b.aid()}] User {U} authenticated to {P2} via SMB from {P1}",
                            "tool_call", [U, P2, P1], 0.3)
        inc["exfil"] = b.add(de + 0.4, f"[{b.aid()}] Large outbound transfer of {gb} GB from {P2} to {Z} "
                             f"over tcp/443", "tool_call", [P2, Z], 0.8)
        inc["outcome"] = rnd.choice(OUTCOMES)
        inc["ctkt"] = inc["tkt"]
        inc["close"] = b.add(de + 1.5, f"[{inc['tkt']}] Ticket closed: {inc['outcome']}. Affected host isolated "
                             f"and credentials rotated.", "agent_note", [inc["tkt"]], 1.0)
        inc["days_fail_to_exfil"] = int(inc["exfil"]["ts"] // DAY - inc["fails"][0]["ts"] // DAY)
        incidents.append(inc)

    # Benign activity touching incident entities (confusers).
    for inc in incidents:
        for _ in range(8):
            d = rnd.uniform(0, 28)
            b.add(d, f"[{b.aid()}] Successful login for user {inc['U']} on vpn-gw-01 from 10.0."
                  f"{rnd.randint(0, 9)}.{rnd.randint(2, 250)}", "tool_call", [inc["U"]])
        for h in (inc["E"], inc["P1"], inc["P2"]):
            for _ in range(4):
                b.add(rnd.uniform(0, 28), rnd.choice([
                    f"[{b.aid()}] Antivirus scan completed on {h} with no findings",
                    f"[{b.aid()}] Patch bundle installed on {h}, reboot completed",
                    f"[{b.aid()}] Backup job for {h} finished successfully"]), "tool_call", [h])

    # Self-service password resets: lexically close to the analyst-requested ones.
    for _ in range(45):
        u = rnd.choice(rest_users)
        b.add(rnd.uniform(0, 28), f"[{b.tid()}] Password reset completed for user {u} via self-service portal",
              "tool_call", [u])

    # Knowledge updates: host roles (2-3 versions) and user departments (2 versions).
    ku = []
    for H in other_hosts[:12]:
        roles = rnd.sample(ROLES, rnd.choice([2, 3]))
        days = sorted(rnd.sample(range(1, 27), len(roles)))
        recs = []
        for j, (role, d) in enumerate(zip(roles, days)):
            text = (f"[CMDB] Host {H} role set to {role}" if j == 0 else
                    f"[CMDB] Host {H} role changed from {roles[j - 1]} to {role}")
            recs.append(b.add(d, text, "agent_note", [H], 0.5))
        ku.append({"kind": "role", "ent": H, "latest": recs[-1], "value": roles[-1]})
    for U in rest_users[:6]:
        d_old, d_new = rnd.sample(DEPTS, 2)
        r1 = b.add(rnd.uniform(1, 8), f"[HR] User {U} department set to {d_old}", "agent_note", [U], 0.4)
        r2 = b.add(rnd.uniform(14, 27), f"[HR] User {U} department changed from {d_old} to {d_new}",
                   "agent_note", [U], 0.5)
        ku.append({"kind": "dept", "ent": U, "latest": r2, "value": d_new})

    # Routine noise.
    noise = [
        "[{a}] Antivirus scan completed on {h} with no findings",
        "[{a}] Blocked inbound port scan from {x} against {h}",
        "[{a}] Failed login for user {u} on {h} from 10.0.{n}.{m} (attempt 1 of 1)",
        "[{a}] Successful login for user {u} on {h} from 10.0.{n}.{m}",
        "[{a}] Patch bundle installed on {h}, reboot completed",
        "[{a}] Backup job for {h} finished successfully",
        "[{a}] DNS query to blocked domain from {h}",
        "[{a}] Service restart on {h}: nginx reloaded",
        "[{a}] TLS certificate renewed for {h}",
        "[{a}] Outbound transfer of {gb} GB from {h} to {x} over tcp/443 (approved backup vendor)",
    ]
    for _ in range(2600):
        t = rnd.choice(noise)
        b.add(rnd.uniform(0, 28.8), t.format(a=b.aid(), h=rnd.choice(hosts), u=rnd.choice(users),
                                              x=rnd.choice(ext[60:100]), n=rnd.randint(0, 9),
                                              m=rnd.randint(2, 250), gb=round(rnd.uniform(0.1, 2.0), 1)),
              "tool_call")

    # Questions.
    qs = []

    def q(qtype, text, answer, gold):
        qs.append({"qtype": qtype, "text": text, "answer": answer, "gold": gold})

    for inc in incidents[:10]:
        q("single-hop", f"What was the outcome of ticket {inc['tkt']}?", inc["outcome"], [inc["close"]])
    for inc in incidents[:8]:
        q("single-hop", f"Which external IP received the large outbound transfer from {inc['P2']}?",
          inc["Z"], [inc["exfil"]])
    for inc in incidents[:12]:
        q("multi-hop", f"Which external IP received data from the host that {inc['U']} reached last via SMB?",
          inc["Z"], [inc["lat2"], inc["exfil"]])
    for inc in incidents[:8]:
        q("multi-hop", f"What was the outcome of the ticket opened for {inc['U']}'s suspected compromise?",
          inc["outcome"], [inc["open"], inc["close"]])
    for inc in incidents[:12]:
        q("temporal", f"Which host did {inc['U']} first access via RDP after the password reset?",
          inc["P1"], [inc["reset"], inc["lat1"]])
    for inc in incidents[:8]:
        q("temporal", f"How many days passed between the first failed login for {inc['U']} from {inc['S']} "
          f"and the large outbound transfer?", str(inc["days_fail_to_exfil"]),
          [inc["fails"][0], inc["exfil"]])
    resets = [i["reset"]["ts"] for i in incidents]
    for inc in incidents:
        n = round((ASKED_AT - inc["reset"]["ts"]) / DAY)
        same_day = [t for t in resets if abs(t - inc["reset"]["ts"]) < DAY and t != inc["reset"]["ts"]]
        if not same_day and n <= 28:
            q("temporal", f"Which user had their password reset at analyst request {n} days ago?",
              inc["U"], [inc["reset"]])
    for k in ku:
        if k["kind"] == "role":
            q("knowledge-update", f"What is the current role of host {k['ent']}?", k["value"], [k["latest"]])
        else:
            q("knowledge-update", f"Which department is user {k['ent']} currently in?", k["value"], [k["latest"]])
    for h in rnd.sample(other_hosts[12:], 8):
        q("abstention", f"Did {h} transfer data to {take_ext()}?", "no record", [])

    recs = sorted(b.recs, key=lambda r: r["ts"])
    for i, r in enumerate(recs):
        r["id"] = f"d{i:05d}"
    questions = []
    for i, x in enumerate(qs):
        questions.append({"id": f"dq{i:03d}", "qtype": x["qtype"], "text": x["text"],
                          "answer": x["answer"], "gold": [g["id"] for g in x["gold"]]})
    return recs, questions


def write(out="datasets/securitydomain", seed=7):
    recs, questions = build(seed)
    Path(out).mkdir(parents=True, exist_ok=True)
    with open(f"{out}/memories.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps({k: r[k] for k in ("id", "text", "ts", "source", "entities", "importance")}) + "\n")
    with open(f"{out}/questions.jsonl", "w") as f:
        for x in questions:
            f.write(json.dumps(x) + "\n")
    return len(recs), questions


if __name__ == "__main__":
    n, qs = write()
    from collections import Counter

    print(n, "memories,", len(qs), "questions", dict(Counter(x["qtype"] for x in qs)))
