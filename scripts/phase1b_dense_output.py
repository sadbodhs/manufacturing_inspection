#!/usr/bin/env python3
"""Phase 1b: send the anomaly map back, or reduce it inside the engine?

A stage-2 model that produces a dense output (a per-pixel anomaly map, or a
reconstructed image) can hand it back in three ways:

  full      the full-resolution output in FP32, as the method's reference code
            returns it; the client computes the score and thresholds the map
  full16    the same output through an FP16 output binding (half the bytes,
            no change to the graph); see the companion repo's model zoo
  reduce    the score and a 1/8-resolution map computed INSIDE the graph: only
            B floats plus B x 1 x S/8 x S/8 leave the engine

Two methods, because their outputs differ by 3x in channels:

  effad_s   EfficientAD-S, whose method already emits a 1-channel map. In
            `reduce` the map is combined at the network's native resolution
            (56 x 56 at 256, padded to 64), the score is its max, and the map is
            resized straight to S/8: the full-resolution upsample never happens.
  unet_r34  U-Net R34 reconstruction (DRAEM-style): a 3-channel image at full
            resolution. In `reduce` the per-pixel squared error against the
            input is taken in the graph, the score is its max, and the map is
            average-pooled to S/8.

Crops of 256 and 512 for EfficientAD-S, 256 for the U-Net; batch 1 and 8.
18 engines, built once, timed in 3 interleaved rounds (scripts/trt_bench.py).

PREDICTIONS (written 2026-09-27, before these engines were built; calibrated
on Phase 1-pre, results/inspection_encoders.tsv)

  P1  For EfficientAD-S the choice barely matters: `reduce` saves <= 3% per crop
      at 256 batch 1, and <= 5% anywhere. Its full map is one channel (0.26 MB
      at 256, D2H 0.014 ms, ~1% of the frame), and skipping the upsample saves
      little compute.
  P2  The FP16 binding halves D2H for both models, but for EfficientAD-S that is
      <= 0.01 ms per crop.
  P3  For the U-Net's 3-channel output, `reduce` removes >= 90% of D2H: about
      0.03 ms per crop, ~4% of the crop's cost at batch 1 and ~10% at batch 8,
      where compute amortises across the batch and transport does not.
  Consequence if these hold: the output-reduction lever belongs to methods
  that emit features or multi-channel images (PatchCore, Phase 1a;
  reconstruction), not to EfficientAD, which already emits a 1-channel map.

Usage (inside the container): python3 phase1b_dense_output.py <work> <raw.tsv> <summary.tsv>
"""
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import trt_bench as tb  # noqa: E402
from inspection_encoders import EfficientADS, unet_r34  # noqa: E402  (same builders as Phase 1-pre)


class EffADReduce(EfficientADS):
    """EfficientAD-S with the score and an S/8 map computed in the graph."""
    def forward(self, x):
        x = (x - self.mean) / self.std
        t = (self.teacher(x) - self.t_mean) / self.t_std
        s = self.student(x)
        ae = self.ae(x)
        m_st = torch.mean((t - s[:, :384]) ** 2, 1, keepdim=True)
        m_ae = torch.mean((ae - s[:, 384:]) ** 2, 1, keepdim=True)
        m_st = 0.1 * (F.pad(m_st, (4, 4, 4, 4)) - self.qa_st) / (self.qb_st - self.qa_st)
        m_ae = 0.1 * (F.pad(m_ae, (4, 4, 4, 4)) - self.qa_ae) / (self.qb_ae - self.qa_ae)
        m = 0.5 * m_st + 0.5 * m_ae                      # native resolution
        score = torch.amax(m, dim=(2, 3))                 # [B, 1]
        small = F.interpolate(m, size=(self.size // 8, self.size // 8), mode="bilinear")
        return score, small


class ReconReduce(torch.nn.Module):
    """Reconstruction -> per-pixel squared error -> score + S/8 map, in the graph."""
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        err = torch.mean((self.net(x) - x) ** 2, 1, keepdim=True)
        return torch.amax(err, dim=(2, 3)), F.avg_pool2d(err, 8)


def exporter(build, size, batch, outs):
    def export(onnx_path):
        m = build().eval()
        with torch.no_grad():
            torch.onnx.export(m, (torch.rand(batch, 3, size, size),), onnx_path,
                              input_names=["images"], output_names=outs,
                              opset_version=17, dynamo=False)
        return sum(p.numel() for p in m.parameters())
    return export


FP16_OUT = ["--outputIOFormats=fp16:chw"]


def engines():
    out = []
    for size in (256, 512):
        for b in (1, 8):
            full = exporter(lambda s=size: EfficientADS(s), size, b, ["anomaly_map"])
            red = exporter(lambda s=size: EffADReduce(s), size, b, ["score", "map_s8"])
            out += [tb.Engine("effad_s_full", size, b, full),
                    tb.Engine("effad_s_full16", size, b, full, flags=FP16_OUT, out_elem_bytes=2),
                    tb.Engine("effad_s_reduce", size, b, red)]
    for b in (1, 8):
        full = exporter(unet_r34, 256, b, ["recon"])
        red = exporter(lambda: ReconReduce(unet_r34()), 256, b, ["score", "map_s8"])
        out += [tb.Engine("unet_r34_full", 256, b, full),
                tb.Engine("unet_r34_full16", 256, b, full, flags=FP16_OUT, out_elem_bytes=2),
                tb.Engine("unet_r34_reduce", 256, b, red)]
    return out


def main():
    work, raw_tsv, sum_tsv = sys.argv[1:4]
    torch.manual_seed(0)
    tb.run(engines(), work, raw_tsv, sum_tsv)


if __name__ == "__main__":
    main()
