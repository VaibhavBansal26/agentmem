"""Input/output guardrails for the investigation agent.

* TopicalRail: refuses to speculate about a person's intent, motives or guilt (rule-based, no model).
* LlamaGuard: Llama Guard 3 safety classifier through a local Ollama server (untested against a real model:
  Ollama was not installed when this was written; the HTTP handling is tested against a fake server).
* Guarded: runs input checks, then the agent, then output checks, and replaces blocked text with a refusal.

Both guards fail closed at run time: an unreachable classifier blocks rather than letting text through.
"""
import json
import re
import urllib.request
from dataclasses import dataclass, field

REFUSAL_INTENT = ("I can report what the records show, but I can't speculate about why a person did something, "
                  "what they intended, or whether they are guilty. Ask me what happened, when, and on which systems.")
REFUSAL_UNSAFE = "I can't help with that request."


@dataclass
class Verdict:
    safe: bool
    guard: str = ""
    reason: str = ""
    categories: list = field(default_factory=list)


INTENT = (r"intend|intent(ion)?s?|intended|intending|motive|motives|motivation|motivated|meant to|trying to|wanted to|"
          r"planning to|planned to|malicious(ly)?|guilty|guilt|innocent|disgruntled|insider threat|lying|lied|"
          r"deliberate(ly)?|on purpose|purposely|knowingly|up to no good|bad actor|has it in for")
FRAMES = (r"do you (think|believe|suspect)|what (was|were) [\w ]{1,30} thinking|why would (he|she|they|\w+) |"
          r"is (he|she|they|\w+) (a )?(malicious|an? insider|guilty|lying|innocent)|who is to blame|"
          r"what do you suspect|your (opinion|guess) (on|about)")
PERSON = (r"\b(he|she|they|him|her|them|his|their|employee|contractor|admin|administrator|analyst|insider|"
          r"user|person|someone|staff|engineer|[a-z]{1,2}[a-z]{3,}\d?)\b")
HANDLE = r"\b[a-z]{4,}\b"
OUT_SPECULATION = (r"(probably|possibly|likely|apparently|clearly|obviously)\s+(wanted|intended|meant|planned|"
                   r"tried|was trying|motivated|malicious|guilty|disgruntled)|appears to (be|have been) "
                   r"(malicious|deliberate|acting deliberately|disgruntled)|seems (malicious|disgruntled|guilty)|"
                   r"(his|her|their) (intent|motive|motivation) (was|were|seems|appears)")


class TopicalRail:
    """Blocks questions about a person's mental state and answers that speculate about it."""

    name = "topical"

    def check_input(self, text):
        t = text.lower()
        has_intent = re.search(rf"\b({INTENT})\b", t) is not None
        has_frame = re.search(FRAMES, t) is not None
        about_person = re.search(PERSON, t) is not None
        # "intent" in a non-human sense (e.g. a ticket's intent) is rare in this domain; require a person cue.
        if (has_intent or has_frame) and about_person:
            return Verdict(False, self.name, "asks about a person's intent, motive or guilt")
        return Verdict(True, self.name)

    def check_output(self, question, answer):
        if re.search(OUT_SPECULATION, answer.lower()):
            return Verdict(False, self.name, "answer speculates about a person's intent or guilt")
        return Verdict(True, self.name)


class LlamaGuard:
    """Llama Guard 3 via Ollama (`ollama pull llama-guard3:1b`). Replies are 'safe' or 'unsafe\\nS1,S2'."""

    name = "llama-guard"

    def __init__(self, url="http://localhost:11434", model="llama-guard3:1b", timeout=30):
        self.url, self.model, self.timeout = url.rstrip("/"), model, timeout

    def available(self):
        try:
            with urllib.request.urlopen(f"{self.url}/api/tags", timeout=3) as r:
                return any(m.get("name", "").startswith(self.model.split(":")[0])
                           for m in json.load(r).get("models", []))
        except Exception:
            return False

    def _classify(self, messages):
        body = json.dumps({"model": self.model, "messages": messages, "stream": False,
                           "options": {"temperature": 0}}).encode()
        req = urllib.request.Request(f"{self.url}/api/chat", body, {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                reply = json.load(r)["message"]["content"].strip()
        except Exception as e:  # fail closed
            return Verdict(False, self.name, f"classifier unavailable: {type(e).__name__}")
        lines = reply.lower().split()
        if lines and lines[0] == "safe":
            return Verdict(True, self.name)
        cats = [c.upper() for c in re.findall(r"s\d+", reply.lower())]
        return Verdict(False, self.name, "classified unsafe", cats)

    def check_input(self, text):
        return self._classify([{"role": "user", "content": text}])

    def check_output(self, question, answer):
        return self._classify([{"role": "user", "content": question}, {"role": "assistant", "content": answer}])


@dataclass
class Guarded:
    """Wrap an async `answer(question) -> text` function with input and output guards."""

    guards: list

    async def __call__(self, question, answer_fn):
        for g in self.guards:
            v = g.check_input(question)
            if not v.safe:
                return refusal(v), v
        text = await answer_fn(question)
        for g in self.guards:
            v = g.check_output(question, text)
            if not v.safe:
                return refusal(v), v
        return text, Verdict(True)


def refusal(v):
    return REFUSAL_INTENT if v.guard == "topical" else REFUSAL_UNSAFE


def build_guards(use_llama_guard=True):
    guards = [TopicalRail()]
    if use_llama_guard:
        lg = LlamaGuard()
        if lg.available():
            guards.append(lg)
    return guards
