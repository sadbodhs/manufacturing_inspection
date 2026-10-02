#!/usr/bin/env python3
"""Phase 5 S1b: how much of the GPU each search method's kernels actually use.

nvidia-smi's "utilization" is the fraction of time any kernel was running; it
reads ~100% for every method running back to back, light or heavy. Nsight
Compute measures each kernel of ONE search call (one 256 crop, deployable
setting) with the worker's NCU=1 mode, and this script folds the kernels into
one row per (method, bank):

  kernel_ms        summed kernel time: the GPU time one crop really costs
  kernels          how many kernels one call launches
  sm_busy_pct      SM throughput, % of peak (time-weighted over the kernels):
                   how hard the compute units work while a kernel runs
  mem_busy_pct     memory throughput, % of peak (time-weighted)
  occupancy_pct    achieved warp occupancy, % of the SMs' maximum (time-weighted)

ncu replays every kernel several times to collect counters, so its durations
come from the replays, not from the timed runs in phase5_speed.tsv.

Measured afterwards: for FAISS and cuVS the capture also took in the worker's
warm-up calls (4x the kernels an Nsight Systems timeline shows for one call), so
kernel_ms and kernels are NOT per-call and are not used. The throughput
percentages are time-weighted averages over identical calls and are unaffected.
Per-call GPU time comes from phase5_nsys.sh instead.

Usage (inside mi-search, GPU otherwise idle): python3 phase5_ncu.py <out.tsv> [methods] [banks]
"""
import csv
import io
import os
import subprocess
import sys

OUT = sys.argv[1]
METHODS = (sys.argv[2] if len(sys.argv) > 2 else "trt_bf,faiss_flat16,faiss_ivf,faiss_ivfpq,cagra").split(",")
BANKS = (sys.argv[3] if len(sys.argv) > 3 else "100000,1000000").split(",")
WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "phase5_worker.py")
MET = {"gpu__time_duration.sum": "dur_ns",
       "sm__throughput.avg.pct_of_peak_sustained_elapsed": "sm",
       "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed": "mem",
       "sm__warps_active.avg.pct_of_peak_sustained_active": "occ"}


def num(s):
    return float(s.replace(",", "")) if s not in ("", "n/a") else 0.0


rows = []
for b in BANKS:
    for m in METHODS:
        cmd = ["ncu", "--profile-from-start", "off", "--metrics", ",".join(MET), "--csv", "--page", "raw",
               sys.executable, WORKER, m, b, "1"]
        p = subprocess.run(cmd, capture_output=True, text=True, env=dict(os.environ, NCU="1"), timeout=3600)
        lines = [l for l in p.stdout.splitlines() if l.startswith('"')]
        if p.returncode != 0 or len(lines) < 3:
            rows.append(dict(method=m, bank=b, note="ncu failed: " + (p.stderr or p.stdout)[-200:].replace("\n", " ")))
            print(m, b, "FAILED", (p.stderr or p.stdout)[-400:], flush=True)
            continue
        r = list(csv.DictReader(io.StringIO("\n".join(lines))))
        r = [x for x in r[1:] if x.get("Kernel Name")]        # row 0 holds units
        k = [{v: num(x[key]) for key, v in MET.items()} for x in r]
        dur = sum(x["dur_ns"] for x in k) or 1.0
        w = lambda key: sum(x[key] * x["dur_ns"] for x in k) / dur  # noqa: E731
        rows.append(dict(method=m, bank=b, kernels=len(k), kernel_ms="%.3f" % (dur / 1e6),
                         sm_busy_pct="%.1f" % w("sm"), mem_busy_pct="%.1f" % w("mem"),
                         occupancy_pct="%.1f" % w("occ"), note=""))
        print(rows[-1], flush=True)

cols = ["method", "bank", "kernels", "kernel_ms", "sm_busy_pct", "mem_busy_pct", "occupancy_pct", "note"]
with open(OUT, "w") as f:
    f.write("\t".join(cols) + "\n")
    for r in rows:
        f.write("\t".join(str(r.get(c, "")) for c in cols) + "\n")
