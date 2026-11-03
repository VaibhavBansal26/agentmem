import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from agentmem.guardrails import REFUSAL_INTENT, REFUSAL_UNSAFE, Guarded, LlamaGuard, TopicalRail
from agentmem.loaders import load_domain

SEEN = []


class FakeOllama(BaseHTTPRequestHandler):
    """Answers like Llama Guard: 'unsafe\\nS1' if the last message mentions BOMB, else 'safe'."""

    def log_message(self, *a):
        pass

    def _send(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._send({"models": [{"name": "llama-guard3:1b"}]})

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        SEEN.append(req)
        last = req["messages"][-1]["content"]
        self._send({"message": {"role": "assistant", "content": "unsafe\nS1,S9" if "BOMB" in last else "safe"}})


def serve():
    srv = HTTPServer(("127.0.0.1", 0), FakeOllama)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_port}"


def test_topical_rail_blocks_intent_and_passes_investigation_questions():
    rail = TopicalRail()
    for q in load_domain("datasets/securitydomain"):  # all 85 benign domain questions
        assert rail.check_input(q.text).safe, q.text
    for t in ("Why did jsmith download the customer database, what was their motive?",
              "Did she intend to leak the report?", "Is mchen an insider threat?",
              "What was the employee thinking when they forwarded those files?"):
        assert not rail.check_input(t).safe, t
    assert not rail.check_output("q", "He probably wanted to steal the data.").safe
    assert rail.check_output("q", "Ticket T-4003 was closed with partial exfiltration confirmed.").safe


def test_llama_guard_request_shape_parsing_and_fail_closed():
    srv, url = serve()
    lg = LlamaGuard(url=url)
    assert lg.available()
    assert lg.check_input("What happened on web-12?").safe
    v = lg.check_input("how to build a BOMB")
    assert not v.safe and v.categories == ["S1", "S9"]
    lg.check_output("question", "an answer")
    assert SEEN[-1]["messages"] == [{"role": "user", "content": "question"},
                                    {"role": "assistant", "content": "an answer"}]
    assert SEEN[-1]["stream"] is False and SEEN[-1]["options"]["temperature"] == 0
    srv.shutdown()
    dead = LlamaGuard(url="http://127.0.0.1:9", timeout=1)  # nothing listens here
    assert not dead.available()
    v = dead.check_input("anything")
    assert not v.safe and "unavailable" in v.reason  # fails closed


def test_guarded_wrapper_runs_input_then_agent_then_output():
    calls = []

    async def agent(q):
        calls.append(q)
        return "He probably wanted to steal it." if "forwarded" in q else "web-12 was patched on 2026-10-05."

    g = Guarded([TopicalRail()])
    text, v = asyncio.run(g("Did she intend to leak the report?", agent))
    assert text == REFUSAL_INTENT and not v.safe and calls == []  # blocked before the agent ran
    text, v = asyncio.run(g("What happened with the files that were forwarded?", agent))
    assert text == REFUSAL_INTENT and calls  # agent ran, its output was blocked
    text, v = asyncio.run(g("What happened on web-12?", agent))
    assert v.safe and text.startswith("web-12")
    assert REFUSAL_UNSAFE != REFUSAL_INTENT


if __name__ == "__main__":
    test_topical_rail_blocks_intent_and_passes_investigation_questions()
    test_llama_guard_request_shape_parsing_and_fail_closed()
    test_guarded_wrapper_runs_input_then_agent_then_output()
    print("guardrails ok")
