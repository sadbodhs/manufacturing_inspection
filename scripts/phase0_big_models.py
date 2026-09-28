#!/usr/bin/env python3
"""Phase 0: do big models gain from batching?

Stage 3 of the inspection pipeline runs large models (an open-vocabulary
detector, SAM to outline) on a fraction of flagged crops. Flagged crops arrive
in bursts, so they COULD be batched. The question is whether that is worth
anything: batch 8 saves yolov8s ~37-39% of its per-frame GPU time, but that
saving comes from filling a GPU that one small frame leaves idle. A model that
already fills the GPU on one image has little idle capacity left to fill.

Models (native input sizes; random weights for the transformers models,
the ultralytics .pt files for the two detectors):

  sam_b_enc   SAM ViT-B image encoder, 1024      (outlining, stage 3)
  sam_h_enc   SAM ViT-H image encoder, 1024      (batch 4 if batch 8 does not fit)
  dinov2_l    DINOv2 ViT-L/14, 518, all tokens   (large foundation backbone)
  rtdetr_l    RT-DETR-L, 640                     (large transformer detector)
  yolov8s     yolov8s, 640                       (the reference: stage 1)

Each at batch 1 and batch 8, built once, timed in 3 interleaved rounds
(scripts/trt_bench.py).

PREDICTIONS (written 2026-09-27, before any of these engines were built;
calibrated on the model zoo's batch-1 rows and the encoder check's savings)

  P1  The per-frame saving from batch 8 falls as batch-1 GPU time rises:
      yolov8s ~38%, RT-DETR-L ~20%, DINOv2-L ~5%, SAM-B ~5%, SAM-H ~2%.
      Above ~10 ms per image, batching saves under 10%.
  P2  SAM-H does not build at batch 8 on 24 GB and falls back to batch 4.
  P3  Weights do not matter for timing: at batch 1 the random-weight SAM-B,
      SAM-H and DINOv2-L land within +/-3% of the model zoo's pretrained
      engines (18.75, 82.64 and 13.65 ms).
  Consequence if P1 holds: stage 3 should be designed for latency (async,
  priority) rather than for batching.

Usage (inside the container): python3 phase0_big_models.py <work> <raw.tsv> <summary.tsv>
"""
import os
import shutil
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import trt_bench as tb  # noqa: E402

MODELS_DIR = "/models"  # triton/models of the benchmark repo, mounted read-only use


class SamEnc(torch.nn.Module):
    """Same wrapper as the model zoo: the image encoder only, first output
    (the 256 x 64 x 64 embedding the mask decoder consumes)."""
    def __init__(self, e):
        super().__init__()
        self.e = e

    def forward(self, x):
        out = self.e(x)
        return out[0] if isinstance(out, (tuple, list)) else out.last_hidden_state


class HFTokens(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return self.m(pixel_values=x).last_hidden_state


def sam(huge):
    from transformers import SamConfig, SamModel, SamVisionConfig
    if huge:
        vc = SamVisionConfig(hidden_size=1280, num_hidden_layers=32, num_attention_heads=16,
                             global_attn_indexes=[7, 15, 23, 31], mlp_dim=5120)
    else:
        vc = SamVisionConfig()  # the defaults are ViT-B: 768 wide, 12 layers, 12 heads
    return SamEnc(SamModel(SamConfig(vision_config=vc.to_dict())).vision_encoder)


def dinov2_l():
    from transformers import Dinov2Config, Dinov2Model
    return HFTokens(Dinov2Model(Dinov2Config(image_size=518, patch_size=14, hidden_size=1024,
                                             num_hidden_layers=24, num_attention_heads=16,
                                             attn_implementation="eager")))


def torch_export(build_fn, size, batch):
    def export(onnx_path):
        m = build_fn().eval()
        with torch.no_grad():
            torch.onnx.export(m, (torch.rand(batch, 3, size, size),), onnx_path,
                              input_names=["images"], output_names=["output0"],
                              opset_version=17, dynamo=False)
        return sum(p.numel() for p in m.parameters())
    return export


def ultra_export(pt_name, size, batch):
    def export(onnx_path):
        from ultralytics import RTDETR, YOLO
        d = os.path.dirname(onnx_path)
        pt = os.path.join(d, pt_name)
        shutil.copy(os.path.join(MODELS_DIR, pt_name), pt)
        m = (RTDETR if pt_name.startswith("rtdetr") else YOLO)(pt)
        params = sum(p.numel() for p in m.model.parameters())
        out = m.export(format="onnx", imgsz=size, batch=batch, dynamic=False, simplify=True)
        shutil.move(out, onnx_path)
        return params
    return export


SPECS = {
    "sam_b_enc": (lambda b: torch_export(lambda: sam(False), 1024, b), 1024, "random"),
    "sam_h_enc": (lambda b: torch_export(lambda: sam(True), 1024, b), 1024, "random"),
    "dinov2_l":  (lambda b: torch_export(dinov2_l, 518, b), 518, "random"),
    "rtdetr_l":  (lambda b: ultra_export("rtdetr-l.pt", 640, b), 640, "pretrained"),
    "yolov8s":   (lambda b: ultra_export("yolov8s.pt", 640, b), 640, "pretrained"),
}


def engine(model, size, batch):
    exp, _, weights = SPECS[model]
    return tb.Engine(model, size, batch, exp(batch), weights,
                     fallback=4 if (model == "sam_h_enc" and batch == 8) else None)


def main():
    work, raw_tsv, sum_tsv = sys.argv[1:4]
    torch.manual_seed(0)
    engines = [engine(m, SPECS[m][1], b) for m in SPECS for b in (1, 8)]
    tb.run(engines, work, raw_tsv, sum_tsv, fallback_factory=engine)


if __name__ == "__main__":
    main()
