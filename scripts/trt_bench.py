#!/usr/bin/env python3
"""Shared engine-timing harness for every phase in this repo.

A phase script describes its engines; this module exports, builds and times
them the same way every time, so numbers from different phases are comparable:

  1. export   each engine's model to ONNX, in its own directory (models over
              2 GB write external weight files next to the .onnx)
  2. build    once, `trtexec --fp16`; the ONNX is deleted after the build
  3. time     REPEATS interleaved rounds: every engine once per round, the
              order rotated each round, trtexec's default timing window
  4. report   raw rows per repeat, then medians per engine

Engine-level numbers (H2D / GPU compute / D2H from trtexec), the same fields
and parsing as the model zoo in sadbodhs/computer_vision_optimization
(scripts/engine_sweep.py), so the two repos' tables can be read side by side.

Runs INSIDE the triton-server container (torch, transformers, ultralytics,
trtexec at /usr/src/tensorrt/bin/trtexec).
"""
import os
import re
import shutil
import statistics
import subprocess

TRTEXEC = "/usr/src/tensorrt/bin/trtexec"
REPEATS = 3


class Engine:
    """One engine to measure.

    export(onnx_path) writes the ONNX and returns the parameter count. If the
    build fails and fallback_batch is set, the phase script is asked for that
    batch instead (e.g. SAM-H at batch 8 -> 4 when it does not fit)."""
    def __init__(self, model, size, batch, export, weights="random", fallback=None):
        self.model, self.size, self.batch = model, size, batch
        self.export, self.weights, self.fallback = export, weights, fallback

    @property
    def tag(self):
        return "%s_%d_b%d" % (self.model, self.size, self.batch)


def _mean(log, label):
    m = re.search(re.escape(label) + r".*?mean = ([\d.]+) ms", log)
    return float(m.group(1)) if m else float("nan")


def _outs(log):
    return re.findall(r"Output binding for \S+ with dimensions ([\dx]+) is created", log)


def _fp32_bytes(dims):
    n = 4
    for d in dims.split("x"):
        n *= int(d)
    return n


def build(e, work):
    d = os.path.join(work, e.tag)
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    onnx_path = os.path.join(d, "model.onnx")
    plan = os.path.join(work, e.tag + ".plan")
    try:
        params = e.export(onnx_path)
    except Exception as ex:  # an export failure is a result, not a crash
        print("%-26s EXPORT_FAILED %s" % (e.tag, str(ex).splitlines()[0][:160]), flush=True)
        shutil.rmtree(d, ignore_errors=True)
        return None
    r = subprocess.run([TRTEXEC, "--onnx=" + onnx_path, "--fp16", "--saveEngine=" + plan],
                       capture_output=True, text=True)
    log = r.stdout + r.stderr
    shutil.rmtree(d, ignore_errors=True)
    if "PASSED" not in log or not os.path.exists(plan):
        tail = [l for l in log.splitlines() if re.search(r"error|Error|failed|memory", l)][-2:]
        print("%-26s BUILD_FAILED %s" % (e.tag, " | ".join(t[:160] for t in tail)), flush=True)
        return None
    print("%-26s built  params=%d" % (e.tag, params), flush=True)
    return params, plan


def run(engines, work, raw_tsv, sum_tsv, fallback_factory=None):
    os.makedirs(work, exist_ok=True)
    built = []
    for e in engines:
        got = build(e, work)
        if got is None and e.fallback and fallback_factory:
            e2 = fallback_factory(e.model, e.size, e.fallback)
            print("%-26s -> falling back to batch %d" % (e.tag, e.fallback), flush=True)
            got, e = build(e2, work), e2
        if got is not None:
            built.append((e, got[0], got[1]))

    rows = {}
    with open(raw_tsv, "w") as raw:
        raw.write("tag\tmodel\tsize\tbatch\tweights\trep\tqps\tgpu_ms\th2d_ms\td2h_ms\tlatency_ms\toutput\n")
        for rep in range(REPEATS):
            k = rep * len(built) // REPEATS
            for e, params, plan in built[k:] + built[:k]:
                r = subprocess.run([TRTEXEC, "--loadEngine=" + plan], capture_output=True, text=True)
                log = r.stdout + r.stderr
                q = re.search(r"Throughput: ([\d.]+) qps", log)
                if "PASSED" not in log or not q:
                    print("%-26s rep%d TIMING_FAILED" % (e.tag, rep + 1), flush=True)
                    continue
                v = dict(qps=float(q.group(1)), gpu=_mean(log, "GPU Compute Time:"),
                         h2d=_mean(log, "H2D Latency:"), d2h=_mean(log, "D2H Latency:"),
                         lat=_mean(log, "Latency:"), out=";".join(_outs(log)))
                rows.setdefault(e.tag, []).append(v)
                raw.write("%s\t%s\t%d\t%d\t%s\t%d\t%.1f\t%.4f\t%.4f\t%.4f\t%.4f\t%s\n"
                          % (e.tag, e.model, e.size, e.batch, e.weights, rep + 1, v["qps"],
                             v["gpu"], v["h2d"], v["d2h"], v["lat"], v["out"]))
                raw.flush()
                print("%-26s rep%d  gpu %8.3f ms  h2d %6.3f  d2h %6.3f"
                      % (e.tag, rep + 1, v["gpu"], v["h2d"], v["d2h"]), flush=True)

    with open(sum_tsv, "w") as f:
        f.write("model\tinput\tbatch\tweights\toutput\tin_bytes\tout_bytes\tparams\treps\tqps"
                "\tgpu_ms\th2d_ms\td2h_ms\tnon_engine_ms\ttransport_pct\tgpu_ms_per_frame"
                "\ttotal_ms_per_frame\tgpu_ms_min\tgpu_ms_max\n")
        for e, params, plan in built:
            vs = rows.get(e.tag, [])
            if not vs:
                continue
            med = {k: statistics.median(v[k] for v in vs) for k in ("qps", "gpu", "h2d", "d2h")}
            out = vs[0]["out"]
            ob = sum(_fp32_bytes(x) for x in out.split(";") if x)
            ne = med["h2d"] + med["d2h"]
            f.write("%s\t%dx3x%dx%d\t%d\t%s\t%s\t%d\t%d\t%d\t%d\t%.1f\t%.4f\t%.4f\t%.4f\t%.4f"
                    "\t%.2f\t%.4f\t%.4f\t%.4f\t%.4f\n"
                    % (e.model, e.batch, e.size, e.size, e.batch, e.weights, out,
                       e.batch * 3 * e.size * e.size * 4, ob, params, len(vs), med["qps"],
                       med["gpu"], med["h2d"], med["d2h"], ne, 100.0 * ne / (med["gpu"] + ne),
                       med["gpu"] / e.batch, (med["gpu"] + ne) / e.batch,
                       min(v["gpu"] for v in vs), max(v["gpu"] for v in vs)))
            os.remove(plan)
