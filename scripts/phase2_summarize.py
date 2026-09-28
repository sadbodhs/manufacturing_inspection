#!/usr/bin/env python3
"""Turn results/phase2_sweep{A,B}.jsonl into per-configuration medians.

Each configuration ran 3 times in an interleaved order; this reports the
median of each metric across the repeats, plus the spread of the fast-path
p99 (min-max), and flags any run that errored or delivered less than 95% of
the offered frames (overloaded: its latencies are backlog, not service time).

Usage: python3 scripts/phase2_summarize.py [A] [B]
Output: results/phase2_sweepA.tsv, results/phase2_sweepB.tsv
"""
import json
import os
import statistics
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KEYS = {"A": ["streams", "k", "batching"],
        "B": ["streams", "p", "priority", "s3_input"]}
METRICS = ["fps", "fast_ms_p50", "fast_ms_p95", "fast_ms_p99", "late_frames",
           "flagged", "s3_sent", "s3_done", "s3_skipped", "explain_ms_p50", "explain_ms_p99"]


def main():
    for sweep in (sys.argv[1:] or ["A", "B"]):
        src = os.path.join(ROOT, "results", "phase2_sweep%s.jsonl" % sweep)
        if not os.path.exists(src):
            continue
        groups, errors = defaultdict(list), 0
        for line in open(src):
            row = json.loads(line)
            if "error" in row["result"]:
                errors += 1
                continue
            groups[tuple(row["config"][k] for k in KEYS[sweep])].append(row["result"])
        out = os.path.join(ROOT, "results", "phase2_sweep%s.tsv" % sweep)
        with open(out, "w") as f:
            f.write("\t".join(KEYS[sweep] + ["reps", "offered_fps"] + METRICS +
                              ["delivered_pct", "overloaded", "skip_pct", "fast_p99_min", "fast_p99_max"]) + "\n")
            for key in sorted(groups):
                rs = groups[key]
                med = {m: statistics.median(r[m] for r in rs) for m in METRICS}
                offered = rs[0]["offered_fps"]
                delivered = 100.0 * med["fps"] / offered
                skip = 100.0 * med["s3_skipped"] / med["flagged"] if med["flagged"] else 0.0
                p99s = [r["fast_ms_p99"] for r in rs]
                f.write("\t".join([str(k) for k in key] + [str(len(rs)), "%.1f" % offered] +
                                  ["%.3f" % med[m] for m in METRICS] +
                                  ["%.1f" % delivered, "yes" if delivered < 95 else "no", "%.1f" % skip,
                                   "%.3f" % min(p99s), "%.3f" % max(p99s)]) + "\n")
        print("sweep %s: %d configurations, %d errored runs -> %s" % (sweep, len(groups), errors, out))


if __name__ == "__main__":
    main()
