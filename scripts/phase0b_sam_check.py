#!/usr/bin/env python3
"""Phase 0b: two things Phase 0 found that must be explained before either is
published.

1. WEIGHTS. This repo times models with random weights on the premise that GPU
   time depends on structure, not learned values. In Phase 0 that held for
   DINOv2-L (+0.8% against the model zoo's pretrained engine) but not for SAM:
   SAM-B -8.3%, SAM-H -9.6%. The two SAM exports differ in more than weights
   (made months apart, from different code paths), so Phase 0 cannot say which
   difference matters. This isolates it on ONE graph: the model zoo's own
   SAM-B ONNX, timed as exported (pretrained) and with every float weight
   replaced by random values of the same mean and spread (randomized). Phase
   0's random-weight export is timed alongside, and the op types of the two
   graphs are counted.

2. BATCHING. SAM-B's encoder is 14.3% SLOWER per image at batch 8 than at 1.
   A layer profile of both engines (trtexec --dumpProfile) shows where.

Inputs: the zoo ONNX, copied to /tmp/mi_inputs/sam_vit_b_enc.onnx by
scripts/phase0b_sam_check.sh.
Outputs: <summary.tsv> (weights timing), plus phase0b_graph_ops.tsv and
phase0b_profile.tsv next to it.

PREDICTIONS (written 2026-09-27, before any of this ran)

  P1  Same graph, pretrained vs randomized: within +/-1%. Weights do not matter;
      the premise holds.
  P2  So the 8-10% gap is the export graph: the zoo's and Phase 0's SAM-B ONNX
      differ in op types (a different attention decomposition), and Phase 0's
      graph is the faster one.
  P3  SAM-B's batch-8 slowdown is concentrated in its 4 global-attention
      blocks, which attend over all 4,096 tokens: >= 50% of the added per-image
      time is in their attention, if the profile resolves layers that finely.

Usage (inside the container): python3 phase0b_sam_check.py <work> <raw.tsv> <summary.tsv>
"""
import collections
import json
import os
import re
import shutil
import subprocess
import sys

import numpy as np
import onnx
import torch
from onnx import numpy_helper

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import trt_bench as tb  # noqa: E402
from phase0_big_models import sam, torch_export  # noqa: E402

ZOO = "/tmp/mi_inputs/sam_vit_b_enc.onnx"


def copy_zoo(onnx_path):
    shutil.copy(ZOO, onnx_path)
    return sum(int(np.prod(i.dims)) for i in onnx.load(ZOO, load_external_data=False).graph.initializer)


def randomized_zoo(onnx_path):
    m = onnx.load(ZOO)
    rng = np.random.default_rng(0)
    n = 0
    for i, init in enumerate(m.graph.initializer):
        a = numpy_helper.to_array(init)
        n += a.size
        if a.dtype.kind == "f" and a.size > 1:
            r = rng.normal(float(a.mean()), float(a.std()) or 0.02, a.shape).astype(a.dtype)
            m.graph.initializer[i].CopyFrom(numpy_helper.from_array(r, init.name))
    onnx.save(m, onnx_path)
    return n


def op_counts(path):
    return collections.Counter(n.op_type for n in onnx.load(path, load_external_data=False).graph.node)


def profile(onnx_path, tag, work):
    """Build, then time with a per-layer profile. Returns [(layer, avg_ms)]."""
    plan = os.path.join(work, tag + ".plan")
    prof = os.path.join(work, tag + ".prof.json")
    r = subprocess.run([tb.TRTEXEC, "--onnx=" + onnx_path, "--fp16", "--saveEngine=" + plan,
                        "--profilingVerbosity=detailed", "--separateProfileRun",
                        "--dumpProfile", "--exportProfile=" + prof],
                       capture_output=True, text=True)
    if not os.path.exists(prof):
        print(tag, "PROFILE_FAILED", (r.stdout + r.stderr)[-300:], flush=True)
        return []
    rows = [x for x in json.load(open(prof)) if "name" in x]
    os.remove(plan)
    return [(x["name"], float(x.get("averageMs", 0.0))) for x in rows]


def category(name):
    n = name.lower()
    for key, pat in (("attention (fused MHA)", r"mha|fmha|attention"),
                     ("softmax", r"softmax"), ("matmul / gemm", r"matmul|gemm|fc"),
                     ("conv", r"conv"), ("norm", r"norm"), ("myelin block", r"foreignnode|myelin"),
                     ("reformat / copy", r"reformat|copy|shuffle|transpose")):
        if re.search(pat, n):
            return key
    return "other"


def main():
    work, raw_tsv, sum_tsv = sys.argv[1:4]
    out_dir = os.path.dirname(sum_tsv)
    torch.manual_seed(0)

    # 1) weights, on one graph
    phase0_export = torch_export(lambda: sam(False), 1024, 1)
    engines = [tb.Engine("sam_b_zoo_graph_pretrained", 1024, 1, copy_zoo, weights="pretrained"),
               tb.Engine("sam_b_zoo_graph_randomized", 1024, 1, randomized_zoo, weights="random"),
               tb.Engine("sam_b_phase0_graph", 1024, 1, phase0_export, weights="random")]
    tb.run(engines, work, raw_tsv, sum_tsv)

    mine = os.path.join(work, "phase0_graph.onnx")
    phase0_export(mine)
    a, b = op_counts(ZOO), op_counts(mine)
    with open(os.path.join(out_dir, "phase0b_graph_ops.tsv"), "w") as f:
        f.write("op_type\tzoo_graph\tphase0_graph\tdiff\n")
        for op in sorted(set(a) | set(b), key=lambda k: -(a[k] + b[k])):
            f.write("%s\t%d\t%d\t%+d\n" % (op, a[op], b[op], b[op] - a[op]))

    # 2) where SAM-B's batch-8 time goes
    per = {}
    for batch in (1, 8):
        p = os.path.join(work, "sam_b_b%d.onnx" % batch)
        torch_export(lambda: sam(False), 1024, batch)(p)
        layers = profile(p, "sam_b_b%d" % batch, work)
        os.remove(p)
        cats = collections.defaultdict(float)
        for name, ms in layers:
            cats[category(name)] += ms / batch          # per image
        per[batch] = (cats, layers)
    with open(os.path.join(out_dir, "phase0b_profile.tsv"), "w") as f:
        f.write("category\tb1_ms_per_image\tb8_ms_per_image\tdelta_ms\n")
        cats = set(per[1][0]) | set(per[8][0])
        for c in sorted(cats, key=lambda c: -(per[8][0][c] - per[1][0][c])):
            f.write("%s\t%.4f\t%.4f\t%+.4f\n" % (c, per[1][0][c], per[8][0][c], per[8][0][c] - per[1][0][c]))
        f.write("#layers\t%d\t%d\t\n" % (len(per[1][1]), len(per[8][1])))
    # the 15 layers that grow most per image, for reading the categories honestly
    with open(os.path.join(out_dir, "phase0b_profile_layers.tsv"), "w") as f:
        f.write("layer\tb1_ms\tb8_ms_per_image\tdelta_ms\n")
        b1 = dict(per[1][1])
        grown = sorted(((n, b1.get(n, 0.0), ms / 8) for n, ms in per[8][1]),
                       key=lambda t: -(t[2] - t[1]))[:15]
        for n, x, y in grown:
            f.write("%s\t%.4f\t%.4f\t%+.4f\n" % (n[:160], x, y, y - x))


if __name__ == "__main__":
    main()
