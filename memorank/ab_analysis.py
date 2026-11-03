"""A/B analysis: python -m memorank.ab_analysis [--db ab/sessions.db]

Decision rule (fixed in ab/PREREG.md before the run): promote B only if each arm has at least
MIN_PER_ARM sessions, the two-sided two-proportion z-test gives p < ALPHA with B > A, AND the
question-clustered bootstrap CI for the difference excludes 0. Sessions repeat the same 85 questions,
so they are not independent; the clustered CI is the guard against over-claiming from the z-test.
"""
import argparse
import math
import random
from collections import defaultdict
from pathlib import Path

from .telemetry import LocalSink

ALPHA, MIN_PER_ARM = 0.05, 150


def z_test(x1, n1, x2, n2):
    """Pooled two-sided two-proportion z-test. Returns (z, p)."""
    p = (x1 + x2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    z = (x2 / n2 - x1 / n1) / se if se else 0.0
    return z, math.erfc(abs(z) / math.sqrt(2))


def cluster_bootstrap(rows, n_boot=2000, seed=0):
    """95% CI for acc(B) - acc(A), resampling whole questions (all their sessions)."""
    by_q = defaultdict(list)
    for r in rows:
        by_q[r["question_id"]].append(r)
    qs, rng, diffs = list(by_q), random.Random(seed), []
    for _ in range(n_boot):
        sample = [r for q in rng.choices(qs, k=len(qs)) for r in by_q[q]]
        a = [r["correct"] for r in sample if r["variant"] == "A"]
        b = [r["correct"] for r in sample if r["variant"] == "B"]
        if a and b:
            diffs.append(sum(b) / len(b) - sum(a) / len(a))
    diffs.sort()
    return diffs[int(0.025 * len(diffs))], diffs[int(0.975 * len(diffs)) - 1]


def analyze(rows):
    arm = {v: [r for r in rows if r["variant"] == v] for v in ("A", "B")}
    n = {v: len(arm[v]) for v in arm}
    x = {v: sum(r["correct"] for r in arm[v]) for v in arm}
    z, p = z_test(x["A"], n["A"], x["B"], n["B"]) if min(n.values()) else (0.0, 1.0)
    lo, hi = cluster_bootstrap(rows) if min(n.values()) else (0.0, 0.0)
    acc = {v: x[v] / n[v] if n[v] else float("nan") for v in arm}
    promote = (min(n.values()) >= MIN_PER_ARM and p < ALPHA and acc["B"] > acc["A"] and lo > 0)
    return dict(n=n, x=x, acc=acc, z=z, p=p, ci=(lo, hi), promote=promote, arm=arm)


def report(db):
    rows = LocalSink(db).rows()
    r = analyze(rows)
    mean = lambda v, k: sum(s[k] for s in r["arm"][v]) / max(len(r["arm"][v]), 1)
    out = ["# Stage 5: online A/B, reranker A (dense top-k) vs B (full pipeline)", "",
           f"Sessions: A={r['n']['A']}, B={r['n']['B']} (pre-registered minimum {MIN_PER_ARM} per arm).", "",
           "| arm | n | accuracy | tool calls | tool-error rows | input tok | output tok | latency s |",
           "|---|---|---|---|---|---|---|---|"]
    for v in ("A", "B"):
        errs = sum(bool(s["tool_errors"]) for s in r["arm"][v])
        out.append(f"| {v} | {r['n'][v]} | {r['acc'][v]:.3f} | {mean(v, 'tool_calls'):.2f} | {errs} | "
                   f"{mean(v, 'input_tokens'):.0f} | {mean(v, 'output_tokens'):.0f} | {mean(v, 'latency_s'):.1f} |")
    clean = [s for s in rows if not s["tool_errors"]]
    ca = {v: [s["correct"] for s in clean if s["variant"] == v] for v in ("A", "B")}
    sens = (f"Sensitivity, excluding sessions with tool errors: A {sum(ca['A']) / max(len(ca['A']), 1):.3f} "
            f"(n={len(ca['A'])}), B {sum(ca['B']) / max(len(ca['B']), 1):.3f} (n={len(ca['B'])}).")
    diff = r["acc"]["B"] - r["acc"]["A"]
    out += ["", f"Difference B - A: {diff:+.3f}. Two-proportion z = {r['z']:.2f}, two-sided p = {r['p']:.4f}.",
            f"Question-clustered bootstrap 95% CI for the difference: [{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}].",
            "", sens, "", f"**Decision under the pre-registered rule: {'PROMOTE B' if r['promote'] else 'DO NOT PROMOTE B'}.**"]
    types = sorted({s["qtype"] for s in rows})
    out += ["", "| type | n(A) | acc A | n(B) | acc B |", "|---|---|---|---|---|"]
    for t in types:
        a = [s["correct"] for s in r["arm"]["A"] if s["qtype"] == t]
        b = [s["correct"] for s in r["arm"]["B"] if s["qtype"] == t]
        out.append(f"| {t} | {len(a)} | {sum(a) / max(len(a), 1):.2f} | {len(b)} | {sum(b) / max(len(b), 1):.2f} |")
    text = "\n".join(out) + "\n"
    Path("results").mkdir(exist_ok=True)
    Path("results/stage5_ab.md").write_text(text)
    return text


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="ab/sessions.db")
    print(report(ap.parse_args().db))
