#!/usr/bin/env python3
"""Phase 2: build the engines the mi-triton server serves.

  s2      WRN50 PatchCore with the in-engine nearest-neighbour search and a
          10k-patch bank (Phase 1a), input 256, DYNAMIC batch 1..16 (opt 4):
          the K crops of a frame arrive as one request, and the dynamic
          batcher may merge several cameras' requests
  s3_*    Grounding DINO-T, 10 phrases, text cached (Phase 1c), batch 1,
          at a 384 crop (s3_crop) and an 800 frame (s3_frame)

Stage 1 is the companion repo's own yolov8s engine, copied, not rebuilt.
Runs inside triton-server (trtexec 10.7, the TensorRT inside Triton 24.12).
Usage: python3 phase2_export.py <out_dir>
"""
import os
import subprocess
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase1a_patchcore import PatchCoreSearch  # noqa: E402
from phase1c_stage3 import GDino  # noqa: E402

TRTEXEC = "/usr/src/tensorrt/bin/trtexec"
OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
torch.manual_seed(0)


def build(onnx_path, plan, extra=()):
    r = subprocess.run([TRTEXEC, "--onnx=" + onnx_path, "--fp16", "--saveEngine=" + plan] + list(extra),
                       capture_output=True, text=True)
    ok = "PASSED" in r.stdout + r.stderr and os.path.exists(plan)
    print(("built " if ok else "BUILD_FAILED ") + plan, flush=True)
    if not ok:
        print((r.stdout + r.stderr)[-1500:])
        sys.exit(1)
    os.remove(onnx_path)


# stage 2: dynamic batch
m = PatchCoreSearch(10000).eval()
p = os.path.join(OUT, "s2.onnx")
with torch.no_grad():
    torch.onnx.export(m, (torch.rand(4, 3, 256, 256),), p, input_names=["images"],
                      output_names=["score", "patch_map"], opset_version=17, dynamo=False,
                      dynamic_axes={"images": {0: "b"}, "score": {0: "b"}, "patch_map": {0: "b"}})
build(p, os.path.join(OUT, "s2.plan"),
      ["--minShapes=images:1x3x256x256", "--optShapes=images:4x3x256x256", "--maxShapes=images:16x3x256x256"])

# stage 3: two input sizes
for name, size in (("s3_crop", 384), ("s3_frame", 800)):
    m = GDino(10, True).eval()
    p = os.path.join(OUT, name + ".onnx")
    with torch.no_grad():
        torch.onnx.export(m, (torch.rand(1, 3, size, size),), p, input_names=["images"],
                          output_names=["logits", "boxes"], opset_version=17, dynamo=False)
    build(p, os.path.join(OUT, name + ".plan"))
