#!/usr/bin/env python3
"""Phase 5 driver: run every (method, bank) worker in a fresh process, sampling the GPU from outside.

For each worker: check the GPU is empty (no compute process; memory at the idle
baseline), read the baseline, start an NVML sampler thread in THIS process
(memory every ~2 ms, utilisation every 100 ms; NVML needs no CUDA context, so
the driver adds nothing to the GPU), run the worker, then cut the samples into
the worker's stage windows. One JSON record per worker run.

Memory is device-wide used minus the idle baseline: what the card has to hold
for this method, every allocator cache included. Utilisation is nvidia-smi's
"GPU utilization": the fraction of time any kernel was running, not how much
of the GPU the kernel used (Phase 5's S1b measures that with Nsight Compute).

Order: 3 repeats, each a fresh shuffle of every (method, bank). The rate sweep
runs in repeat 1 only.

Usage (inside mi-search): python3 phase5_driver.py <out.jsonl> [methods] [banks] [reps]
"""
import json
import os
import random
import statistics
import subprocess
import sys
import threading
import time

import pynvml

OUT = sys.argv[1]
METHODS = (sys.argv[2] if len(sys.argv) > 2 else
           "trt_bf,trt_bf512,faiss_flat16,faiss_ivf,faiss_ivfpq,cagra,cpu_flat,cpu_ivf").split(",")
BANKS = [int(b) for b in (sys.argv[3] if len(sys.argv) > 3 else "10000,100000,1000000").split(",")]
REPS = int(sys.argv[4]) if len(sys.argv) > 4 else 3
WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "phase5_worker.py")

pynvml.nvmlInit()
H = pynvml.nvmlDeviceGetHandleByIndex(0)
MB = 2 ** 20


def used_mb():
    return pynvml.nvmlDeviceGetMemoryInfo(H).used / MB


def gpu_idle():
    return not pynvml.nvmlDeviceGetComputeRunningProcesses(H)


class Sampler(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.mem, self.util, self.stop = [], [], False

    def run(self):
        last_u = 0.0
        while not self.stop:
            t = time.monotonic()
            self.mem.append((t, used_mb()))
            if t - last_u >= 0.1:
                self.util.append((t, pynvml.nvmlDeviceGetUtilizationRates(H).gpu))
                last_u = t
            time.sleep(0.002)


def mem_at(s, t, base):
    xs = [m for (ts, m) in s.mem if t - 0.05 <= ts <= t]
    return (max(xs) if xs else max(m for (ts, m) in s.mem if ts <= t)) - base


def peak(s, t0, t1, base):
    xs = [m for (ts, m) in s.mem if t0 <= ts <= t1]
    return (max(xs) - base) if xs else None


def util(s, t0, t1):
    xs = [u for (ts, u) in s.util if t0 + 0.5 <= ts <= t1]   # skip NVML's averaging lag
    return statistics.mean(xs) if xs else None


def run_one(method, bank, rep):
    for _ in range(120):
        if gpu_idle():
            break
        time.sleep(1)
    base = statistics.median(used_mb() for _ in range(50))
    s = Sampler(); s.start(); time.sleep(0.3)
    t_start = time.monotonic()
    p = subprocess.run([sys.executable, WORKER, method, str(bank), str(rep)],
                       capture_output=True, text=True, timeout=3600)
    time.sleep(0.3); s.stop = True; s.join()
    marks = [json.loads(l) for l in p.stdout.splitlines() if l.startswith("{")]
    rec = dict(method=method, bank=bank, rep=rep, rc=p.returncode, base_mb=base,
               load1=os.getloadavg()[0], wall_s=time.monotonic() - t_start)
    if p.returncode != 0:
        rec["error"] = p.stderr[-1500:]
    m = {k["stage"]: k for k in marks}
    if "ctx" in m:
        rec["ctx_mb"] = mem_at(s, m["ctx"]["t"], base)
    if "lib" in m:
        rec["lib_mb"] = mem_at(s, m["lib"]["t"], base)
    if "built" in m:
        rec["build_peak_mb"] = peak(s, m["lib"]["t"], m["built"]["t"], base)
        rec.update({k: v for k, v in m["built"].items() if k not in ("stage", "t")})
    if "host_freed" in m:
        rec["resident_mb"] = mem_at(s, m["host_freed"]["t"], base)
        rec["rss_after_mb"] = m["host_freed"]["rss_mb"]
    rec["process_peak_mb"] = peak(s, t_start, time.monotonic(), base)
    win = {}
    for k in marks:
        st = k["stage"]
        if st.endswith("_begin"):
            win[(st[:-6], k["tag"], k.get("frac"))] = k
        elif st.endswith("_end"):
            b = win.pop((st[:-4], k["tag"], k.get("frac")))
            row = dict(kind=st[:-4], tag=k["tag"], frac=k.get("frac"),
                       peak_mb=peak(s, b["t"], k["t"], base), util=util(s, b["t"], k["t"]))
            row.update({x: v for x, v in k.items() if x not in ("stage", "t", "tag", "frac")})
            rec.setdefault("windows", []).append(row)
    rec["recall"] = [{x: v for x, v in k.items() if x not in ("stage", "t")} for k in marks if k["stage"] == "recall"]
    return rec


def main():
    cfgs = [(m, b) for b in BANKS for m in METHODS]
    for rep in range(1, REPS + 1):
        order = cfgs[:]
        random.Random(100 + rep).shuffle(order)
        for method, bank in order:
            rec = run_one(method, bank, rep)
            with open(OUT, "a") as f:
                f.write(json.dumps(rec) + "\n")
            w = {x["tag"]: round(x.get("ms", 0), 3) for x in rec.get("windows", []) if x["kind"] == "lat"}
            print(rep, method, bank, "rc=%d" % rec["rc"], "resident=%s" % round(rec.get("resident_mb") or -1),
                  "peak=%s" % round(rec.get("process_peak_mb") or -1), w, flush=True)


if __name__ == "__main__":
    main()
