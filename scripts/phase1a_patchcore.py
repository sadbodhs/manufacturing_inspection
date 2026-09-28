#!/usr/bin/env python3
"""Phase 1a: PatchCore's cost, split into backbone and memory-bank search.

PatchCore keeps a memory bank of patch features from good parts. At inference
every patch of the crop is compared with every bank entry, and a patch's
anomaly score is the distance to its nearest neighbour. Phase 1-pre measured
the backbone (WRN50-2 + feature head) and found its output heavy to move:
1536 x 32 x 32 at 256 (6.3 MB), 1536 x 64 x 64 at 512 (25 MB, 0.96 ms of D2H).

Here the search moves INTO the engine, so only a score and a patch-level map
leave it:

  d^2(patch, bank) = |f|^2 - 2 f.B^T + |B|^2        one matrix multiply
  patch score      = min over the bank              one reduction
  image score      = max over patches;  map = patch scores at 1/8 resolution

The bank is a graph constant (random, standing in for a coreset of good-part
patches). Real banks are a coreset of 1-10% of all training patches: 200 good
images at 256 give ~205k patches, so 1k / 10k / 100k spans a small fixture to a
large one.

Engines: backbone only (the Phase 1-pre reference, re-measured in this run) and
backbone + search with banks of 1k / 10k / 100k, at crops of 256 and 512 batch 1
and 256 batch 8. 12 engines, built once, timed in 3 interleaved rounds.

PREDICTIONS (written 2026-09-27, before these engines were built)

  P1  The search is a matmul of (patches x 1536) by (1536 x bank). At 256
      (1,024 patches) it adds ~0.1 ms at 1k, ~0.7 ms at 10k and ~6 ms at 100k.
      At 512 (4,096 patches) about 4x that: ~25 ms at 100k.
  P2  With a 1k or 10k bank, backbone + in-graph search costs LESS per crop than
      backbone + sending the features out, at 512: the 0.96 ms of D2H it removes
      exceeds the search it adds. At 100k the search dominates everything.
  P3  So PatchCore's cost is set by bank size, not by the backbone, once the bank
      passes ~10k; and at 100k it is slower than EfficientAD-S (1.0 ms at 256)
      by an order of magnitude.
  P4  Batch 8 barely helps the search (it is already one large matmul); the
      per-crop saving at 100k is < 10%.

Fallback if the in-graph search fails to build or does not fit: FAISS on the GPU.

Usage (inside the container): python3 phase1a_patchcore.py <work> <raw.tsv> <summary.tsv>
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import trt_bench as tb  # noqa: E402
from inspection_encoders import PatchCoreWRN50  # noqa: E402  (same backbone as Phase 1-pre)


class PatchCoreSearch(torch.nn.Module):
    def __init__(self, bank_size):
        super().__init__()
        self.backbone = PatchCoreWRN50()
        bank = torch.randn(bank_size, 1536)
        self.register_buffer("bank_t", bank.t().contiguous())            # 1536 x M
        self.register_buffer("bank_sq", (bank ** 2).sum(1).view(1, 1, -1))  # 1 x 1 x M

    def forward(self, x):
        f = self.backbone(x)                           # B x 1536 x h x w
        b, c, h, w = f.shape
        f = f.flatten(2).transpose(1, 2)               # B x hw x 1536
        d2 = (f ** 2).sum(-1, keepdim=True) - 2 * torch.matmul(f, self.bank_t) + self.bank_sq
        patch = torch.sqrt(torch.clamp(torch.amin(d2, -1), min=0))   # B x hw
        return torch.amax(patch, 1, keepdim=True), patch.view(b, 1, h, w)


def exporter(build, size, batch, outs):
    def export(onnx_path):
        m = build().eval()
        with torch.no_grad():
            torch.onnx.export(m, (torch.rand(batch, 3, size, size),), onnx_path,
                              input_names=["images"], output_names=outs,
                              opset_version=17, dynamo=False)
        return sum(p.numel() for p in m.parameters())
    return export


def engines():
    out = []
    for size, b in ((256, 1), (512, 1), (256, 8)):
        out.append(tb.Engine("wrn50_pc_backbone", size, b,
                             exporter(PatchCoreWRN50, size, b, ["features"])))
        for m in (1000, 10000, 100000):
            out.append(tb.Engine("wrn50_pc_search%dk" % (m // 1000), size, b,
                                 exporter(lambda m=m: PatchCoreSearch(m), size, b, ["score", "patch_map"])))
    return out


def main():
    work, raw_tsv, sum_tsv = sys.argv[1:4]
    torch.manual_seed(0)
    tb.run(engines(), work, raw_tsv, sum_tsv)


if __name__ == "__main__":
    main()
