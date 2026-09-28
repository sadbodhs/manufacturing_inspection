#!/usr/bin/env python3
"""Phase 2 sweeps: the whole inspection line under live camera load.

Runs cpp/inspect_client inside the mi-triton container once per configuration
and repeat, in a seeded interleaved order (every configuration once per round,
shuffled per round), with a pause between runs so no run inherits the previous
one's queue. Each run appends one JSON line to results/phase2_<sweep>.jsonl;
scripts/phase2_summarize.py turns that into medians.

  A  do crops batch well?  p = 0, stage 3 off
     cameras 1/4/8/16 x K = 1/4/16 x stage-2 batching off vs dynamic (0 us)
     x 3 repeats = 72 runs
  B  does the rare heavy stage slow the fast path?  K = 4
     cameras 4/8 x p = 0/1/5/20% x stage 1-2 priority off/on
     x stage-3 input crop (384) vs frame (800) x 3 repeats = 96 runs

Model names in the mi-triton repository (scripts/phase2_setup.sh):
  yolov8s, yolov8s_prio            stage 1 (prio: PRIORITY_MAX)
  s2, s2_db0, s2_prio              stage 2 (db0: dynamic batching, 0 us window)
  s3_crop, s3_frame                stage 3 at 384 / 800

Run on the host, under the GPU lock (scripts/phase2_run.sh holds it).
Usage: python3 scripts/phase2_sweep.py A|B|C [--duration 15] [--repeats 3]

PREDICTIONS (written 2026-09-28, before the server or client had run; from the
engine rows of Phases 1a and 1c)

GPU per frame = yolov8s 1.0 ms + stage 2 at batch K. Interpolating Phase 1a's
10k-bank search (0.92 ms at batch 1, 4.98 ms at batch 8): K=1 ~0.9, K=4 ~2.7,
K=16 ~9.6 ms, so ~1.9 / 3.7 / 10.6 ms per frame. At 30 fps a camera offers
33.3 ms per frame, so a 3090 holds roughly 17 / 9 / 3 cameras.

  A1  Capacity follows that arithmetic within ~20%: K=4 sustains 8 cameras
      (p99 fast path < 33 ms, no late frames) and is overloaded at 16; K=16 is
      overloaded from 4 cameras; K=1 holds 16 but near saturation.
  A2  Fast path at 1 camera, p50: ~3 ms (K=1), ~5 ms (K=4), ~11 ms (K=16).
  A3  The dynamic batcher (0 us window) changes little when each request is
      already a batch of K >= 4 (p99 within 10%). It helps most at K=1 with
      8-16 cameras, where it can merge requests near saturation.

  B1  Stage 3 is the load, not the fast path: K*p flagged crops per frame at
      ~9.5 ms (384 crop) or ~29.6 ms (800 frame) each. At p=20% with 4
      cameras that is ~30 ms (crop) of stage-3 GPU per 33 ms of wall time on
      top of ~15 ms of fast path: the GPU is over-committed and stage 3 starts
      skipping crops (all slots busy).
  B2  Without priority, the fast path's p99 rises with p: at p=5% (crop, 4
      cameras) it is >= 50% higher than at p=0, because a 10-30 ms stage-3
      execution holds the GPU while fast-path requests wait.
  B3  Priority (PRIORITY_MAX on stages 1-2) recovers less than half of that
      rise: CUDA stream priority reorders which kernels start next but cannot
      preempt one that is running, and Grounding DINO's kernels are long.
      (First check: does Triton 24.12 turn the setting into stream priority?)
  B4  Frame input (800) costs ~3x a crop (384), so at the same p it skips
      ~3x as often and hurts the fast path more.

PHASE 3, SWEEP C (written 2026-09-28, after sweeps A and B, before C ran)

Sweep B showed the slot pool does not protect the line: at 8 cameras no slot
filled and the line collapsed anyway. C gives stage 3 a GPU-time budget (a
token bucket charged with Phase 1c's measured cost: 9.43 ms a crop, 29.60 ms a
frame); flags beyond it are shed and counted. K = 4, priority off; cameras
4/8 x p 5/20% x crop/frame x budget none/10/20/40% x 3 repeats = 96 runs.
From sweep A the fast path alone uses ~41% of the GPU at 4 cameras and ~82% at 8.

  C1  At 8 cameras a 10% budget prevents the collapse at every p and input:
      >= 95% of frames delivered, where sweep B collapsed at p >= 5%.
  C2  Budgets above the headroom bring it back: at 8 cameras 20% is at the edge
      and 40% collapses; at 4 cameras every budget holds.
  C3  A budget fixes the collapse, not the tail: whenever frames are admitted
      (800 input), fast-path p99 stays >= ~30 ms at 4 cameras, one stage-3
      execution above the no-stage-3 p99, whatever the budget.
  C4  Stage 3 is served at ~budget/cost: ~10.6/s crops or ~3.4/s frames per
      10% (within 15%), and explain latency falls from seconds to tens of ms.
"""
import argparse
import itertools
import json
import os
import random
import subprocess
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLIENT = ["docker", "exec", "mi-triton", "/mi/cpp/build/inspect_client",
          "--url", "localhost:8101", "--file", "/mi/data/frames.bin", "--fps", "30"]


def configs(sweep):
    if sweep == "A":
        for cams, k, db in itertools.product((1, 4, 8, 16), (1, 4, 16), ("off", "db0")):
            yield dict(streams=cams, k=k, batching=db, stage1="yolov8s",
                       stage2="s2_db0" if db == "db0" else "s2", stage3="none", p=0.0)
    elif sweep == "C":
        # Phase 3: stage 3 under a GPU-time budget. Costs are Phase 1c's measured
        # GPU medians for Grounding DINO-T, 10 phrases, text cached.
        for cams, p, s3in, budget in itertools.product((4, 8), (0.05, 0.20), ("crop", "frame"),
                                                       (0.0, 0.10, 0.20, 0.40)):
            yield dict(streams=cams, k=4, p=p, priority="off", s3_input=s3in, budget=budget,
                       stage1="yolov8s", stage2="s2",
                       stage3="s3_crop" if s3in == "crop" else "s3_frame",
                       s3_size=384 if s3in == "crop" else 800,
                       s3_cost_ms=9.43 if s3in == "crop" else 29.60)
    else:
        for cams, p, prio, s3in in itertools.product((4, 8), (0.0, 0.01, 0.05, 0.20), ("off", "on"),
                                                     ("crop", "frame")):
            yield dict(streams=cams, k=4, p=p, priority=prio, s3_input=s3in,
                       stage1="yolov8s_prio" if prio == "on" else "yolov8s",
                       stage2="s2_prio" if prio == "on" else "s2",
                       stage3="s3_crop" if s3in == "crop" else "s3_frame",
                       s3_size=384 if s3in == "crop" else 800)


def clear_shm():
    """Safety net between runs: drop any CUDA shm registration a previous client
    left behind (the client unregisters its own; a crashed one would not)."""
    import urllib.request
    req = urllib.request.Request("http://localhost:8100/v2/cudasharedmemory/unregister", method="POST", data=b"")
    urllib.request.urlopen(req, timeout=10).read()


def run_one(c, duration, seed):
    clear_shm()
    cmd = CLIENT + ["--streams", str(c["streams"]), "--k", str(c["k"]), "--s2-size", "256",
                    "--stage1", c["stage1"], "--stage2", c["stage2"], "--stage3", c["stage3"],
                    "--p", str(c["p"]), "--duration", str(duration), "--seed", str(seed)]
    if c["stage3"] != "none":
        cmd += ["--s3-input", c["s3_input"], "--s3-size", str(c["s3_size"])]
    if c.get("budget"):
        cmd += ["--s3-budget", str(c["budget"]), "--s3-cost-ms", str(c["s3_cost_ms"])]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=duration + 120)
    line = [l for l in r.stdout.splitlines() if l.startswith("{")]
    if r.returncode != 0 or not line:
        return {"error": (r.stderr or r.stdout)[-400:], "returncode": r.returncode}
    return json.loads(line[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep", choices=["A", "B", "C"])
    ap.add_argument("--duration", type=float, default=15)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--pause", type=float, default=3)
    a = ap.parse_args()
    out = os.path.join(ROOT, "results", "phase2_sweep%s.jsonl" % a.sweep)
    cs = list(configs(a.sweep))
    rng = random.Random(20260927)
    n, total = 0, len(cs) * a.repeats
    with open(out, "a") as f:
        for rep in range(1, a.repeats + 1):
            order = cs[:]
            rng.shuffle(order)
            for c in order:
                n += 1
                res = run_one(c, a.duration, seed=1)   # same seed: same crops/flags for every config
                row = dict(config=c, rep=rep, result=res, t=time.strftime("%Y-%m-%dT%H:%M:%S"))
                f.write(json.dumps(row) + "\n")
                f.flush()
                r = res
                print("[%d/%d] rep%d %s -> %s" % (
                    n, total, rep, " ".join("%s=%s" % kv for kv in c.items() if kv[0] not in ("stage1", "stage2", "stage3")),
                    "ERROR " + r["error"][-120:] if "error" in r else
                    "fps %.1f fast p50 %.2f p99 %.2f late %d s3 %d/%d skip %d shed %d explain p50 %.1f"
                    % (r["fps"], r["fast_ms_p50"], r["fast_ms_p99"], r["late_frames"],
                       r["s3_done"], r["s3_sent"], r["s3_skipped"], r.get("s3_shed", 0),
                       r["explain_ms_p50"])), flush=True)
                time.sleep(a.pause)


if __name__ == "__main__":
    main()
