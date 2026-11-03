import math

from agentmem.ab_analysis import analyze, cluster_bootstrap, z_test
from agentmem.flags import Flags
from agentmem.telemetry import LocalSink


def test_z_test_matches_hand_calculation():
    z, p = z_test(40, 100, 60, 100)  # p-hat .5, se = sqrt(.25 * .02) -> z = .2 / .0707
    assert math.isclose(z, 2.8284, abs_tol=1e-3) and math.isclose(p, 0.00468, abs_tol=1e-4)
    assert z_test(50, 100, 50, 100)[1] == 1.0


def test_flags_are_sticky_deterministic_and_balanced():
    f = Flags()
    f.set("reranker", {"A": 0.5, "B": 0.5})
    first = {f"s{i}": f.variant("reranker", f"s{i}") for i in range(2000)}
    assert all(f.variant("reranker", s) == v for s, v in first.items())  # sticky
    g = Flags()
    g.set("reranker", {"A": 0.5, "B": 0.5})
    assert all(g.variant("reranker", s) == v for s, v in first.items())  # same hash in a fresh store
    share_b = sum(v == "B" for v in first.values()) / len(first)
    assert 0.45 < share_b < 0.55
    h = Flags()
    h.set("x", {"A": 0.9, "B": 0.1})
    assert sum(h.variant("x", f"s{i}") == "B" for i in range(2000)) / 2000 < 0.15  # weights respected


def test_decision_rule_requires_sample_size_and_clustered_ci():
    def rows(nA, okA, nB, okB):
        out = []
        for i in range(nA):
            out.append(dict(variant="A", correct=int(i < okA), question_id=f"q{i % 85}", qtype="t"))
        for i in range(nB):
            out.append(dict(variant="B", correct=int(i < okB), question_id=f"q{i % 85}", qtype="t"))
        return out

    assert not analyze(rows(40, 20, 40, 36))["promote"]  # big gap, but below the pre-registered n
    big = analyze(rows(150, 90, 150, 120))
    assert big["p"] < 0.05 and big["promote"]
    assert not analyze(rows(150, 100, 150, 104))["promote"]  # not significant
    lo, hi = cluster_bootstrap(rows(150, 90, 150, 120))
    assert lo > 0 > -hi


def test_local_sink_roundtrip_and_resume(tmp_path):
    s = LocalSink(str(tmp_path / "s.db"))
    row = dict(session_id="s0000", variant="B", question_id="dq000", qtype="single-hop", correct=1,
               tool_calls=2, tool_errors=0, input_tokens=100, output_tokens=20, latency_s=1.5,
               model="m", answer="a")
    s.record(**row)
    s.record(**row)  # idempotent per session id
    assert s.done() == {"s0000"} and len(s.rows()) == 1


if __name__ == "__main__":
    import pathlib
    import tempfile

    test_z_test_matches_hand_calculation()
    test_flags_are_sticky_deterministic_and_balanced()
    test_decision_rule_requires_sample_size_and_clustered_ci()
    test_local_sink_roundtrip_and_resume(pathlib.Path(tempfile.mkdtemp()))
    print("stage5 ok")
