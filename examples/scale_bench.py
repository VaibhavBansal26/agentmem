"""Recall latency vs store size, isolating pipeline overhead: python examples/scale_bench.py [--profile N]

Uses a cheap NumPy embedder and no cross-encoder. The cross-encoder scores a fixed pool of about 40 candidates
per query regardless of store size, so it adds a roughly constant cost that this benchmark leaves out.
"""
import argparse
import random
import statistics
import time
import zlib

import numpy as np

from memorank import Memory

KINDS = ["web", "db", "app", "wks", "srv", "fs"]
TEMPLATES = ["Antivirus scan completed on {h} with no findings", "Backup job for {h} finished successfully",
             "Failed login for user {u} on {h} from 10.0.{a}.{b}", "Patch bundle installed on {h}, reboot completed",
             "Blocked inbound port scan from 203.0.113.{a} against {h}", "TLS certificate renewed for {h}",
             "Host {h} role changed from staging web server to production web server",
             "Outbound transfer of {a}.{b} GB from {h} to 198.51.100.{a} over tcp/443"]


def bow(texts, dim=96):
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        for w in t.lower().split():
            out[i, zlib.crc32(w.encode()) % dim] += 1
    return out


def build(n, seed=0):
    rnd = random.Random(seed)
    mem = Memory(embedder=bow, rerank=True, cross_encoder=False, now="2027-01-01")
    recs = []
    for i in range(n):
        h = f"{rnd.choice(KINDS)}-{rnd.randint(1, max(10, n // 12)):04d}"
        recs.append({"text": rnd.choice(TEMPLATES).format(h=h, u=f"user{rnd.randint(1, 500)}", a=rnd.randint(2, 250),
                                                           b=rnd.randint(2, 250)),
                     "timestamp": 1790812800 + i * 60})
    t = time.time()
    mem.add_many(recs)
    return mem, time.time() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="*", default=[1000, 3000, 10000, 30000])
    ap.add_argument("--queries", type=int, default=12)
    ap.add_argument("--profile", type=int, default=0, help="cProfile one store of this size instead")
    a = ap.parse_args()
    queries = [f"What is the current role of {k}-{n:04d}?" for k, n in zip(KINDS * 4, range(5, 400, 17))][:a.queries]
    if a.profile:
        import cProfile
        import pstats

        mem, _ = build(a.profile)
        mem.recall(queries[0])  # warm: first call embeds every memory
        pr = cProfile.Profile()
        pr.enable()
        for q in queries:
            mem.recall(q, k=8)
        pr.disable()
        pstats.Stats(pr).sort_stats("cumulative").print_stats(14)
        return
    print("| memories | load s | first recall s (embeds all) | median recall ms | p95 ms |\n|---|---|---|---|---|")
    for n in a.sizes:
        mem, load = build(n)
        t = time.time()
        mem.recall(queries[0], k=8)
        first = time.time() - t
        lat = []
        for q in queries:
            t = time.time()
            mem.recall(q, k=8)
            lat.append((time.time() - t) * 1000)
        lat.sort()
        print(f"| {n:,} | {load:.1f} | {first:.1f} | {statistics.median(lat):.0f} | {lat[int(len(lat) * 0.95) - 1]:.0f} |")


if __name__ == "__main__":
    main()
