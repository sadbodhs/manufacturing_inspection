#!/usr/bin/env python3
"""Manufacturing inspection, Phase 1-pre: how fast are the stage-2 encoders?

Stage 2 of the inspection pipeline scores every part crop with an anomaly model
trained on good parts only. This picks which encoders Phases 1-2 carry forward,
by measuring their TensorRT FP16 engine cost on the 3090. Inference only: every
model has RANDOM weights, because GPU time depends on the network's structure,
not on what it learned (and random weights mean nothing is downloaded).

Candidates, each exported the way its method consumes it:

  wrn50_pc   WideResNet-50-2 stem..layer3 + PatchCore feature head in the graph:
             3x3 avg-pool on layer2 and layer3, layer3 upsampled to layer2's
             grid, concatenated -> 1536 channels at 1/8 resolution. (PatchCore,
             PaDiM; the anomalib default.)
  effad_s    EfficientAD-S: teacher PDN (384 out) + student PDN (768 out) + the
             small autoencoder, with the anomaly map combined in the graph
             (st and ae maps, quantile-normalised, padded, upsampled to the input).
  dinov2_s   DINOv2 ViT-S/14, patch tokens (CLS dropped).  (AnomalyDINO)
  dinov2_b   DINOv2 ViT-B/14, patch tokens.
  r18_l123   ResNet-18 stem..layer3, all three feature maps out.  (PaDiM, STFPM,
             FastFlow light; the cheap floor.)
  unet_r34   U-Net with a ResNet-34 encoder (smp), 3-channel output.  (DRAEM-style
             reconstruction; REFERENCE ONLY, so the others read against U-Net.)

Sizes: CNNs at crops of 256 and 512, ViTs (patch 14) at 252 and 504; batch 1 at
both sizes, batch 8 at the small one. 18 engines.

Method: each engine is built ONCE (trtexec --fp16), then timed in 3 interleaved
rounds (every engine once per round, order rotated each round) with trtexec's
default 3 s timing window. Raw rows per repeat, then medians. Engine-level
numbers, comparable to results/v3/model_zoo.tsv (same trtexec, same defaults).

PREDICTIONS (written 2026-09-27, before any of these engines existed; calibrated
on model_zoo.tsv: resnet50@224 0.44 ms, unet_r34@640 1.82 ms, dinov2_l@518 13.65 ms)

  P1  r18_l123 is the fastest: ~0.15 ms GPU at 256, ~0.45 ms at 512.
  P2  wrn50_pc ~1.1 ms at 256, ~4.2 ms at 512. Its 1536x32x32 FP32 output
      (6.3 MB) costs ~0.25 ms D2H, so transport is ~20% of its frame at 256.
  P3  effad_s is NOT the cheapest network. ~38 GMAC at 256 (3.4x wrn50_pc's
      backbone) -> ~2 ms at 256, ~8.5 ms at 512: SLOWER than the WRN50 backbone.
      Its real-time claim comes from skipping PatchCore's nearest-neighbour
      search, not from a lighter network.
  P4  dinov2_s ~0.4 ms at 252, ~1.3 ms at 504; dinov2_b ~1.0 ms at 252,
      ~3.9 ms at 504. So ViT-S at 252 is within reach (cheaper than wrn50_pc).
  P5  unet_r34 ~0.35 ms at 256, ~1.2 ms at 512.
  P6  Batch 8 saves most per frame where batch 1 is launch-bound: r18_l123 and
      dinov2_s ~40%, unet_r34 ~25%, wrn50_pc ~15%, effad_s <10%.
  P7  r18_l123's three feature maps (1.84 MB) make transport ~40% of its frame:
      for the cheapest encoder, moving features costs as much as computing them.

Runs INSIDE the triton-server container (torch, transformers, smp, trtexec).
Usage: python3 inspection_encoders.py <work_dir> <raw.tsv> <summary.tsv> [model ...]
"""
import os
import re
import statistics
import subprocess
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

TRTEXEC = "/usr/src/tensorrt/bin/trtexec"
REPEATS = 3
IMNET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMNET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# --------------------------------------------------------------------------
# Builders (all random weights)
# --------------------------------------------------------------------------
class PatchCoreWRN50(nn.Module):
    def __init__(self):
        super().__init__()
        import torchvision.models as M
        m = M.wide_resnet50_2(weights=None)
        self.stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool)
        self.layer1, self.layer2, self.layer3 = m.layer1, m.layer2, m.layer3
        self.pool = nn.AvgPool2d(3, 1, 1)

    def forward(self, x):
        f2 = self.layer2(self.layer1(self.stem(x)))
        f3 = self.layer3(f2)
        f2, f3 = self.pool(f2), self.pool(f3)
        f3 = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat([f2, f3], 1)


class PDNSmall(nn.Module):
    """EfficientAD's small patch description network (no padding, as in anomalib)."""
    def __init__(self, out_ch):
        super().__init__()
        self.c1, self.c2 = nn.Conv2d(3, 128, 4), nn.Conv2d(128, 256, 4)
        self.c3, self.c4 = nn.Conv2d(256, 256, 3), nn.Conv2d(256, out_ch, 4)
        self.pool = nn.AvgPool2d(2, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.c1(x)))
        x = self.pool(F.relu(self.c2(x)))
        return self.c4(F.relu(self.c3(x)))


class EffADAutoencoder(nn.Module):
    def __init__(self, size):
        super().__init__()
        s = size
        self.enc = nn.ModuleList([nn.Conv2d(3, 32, 4, 2, 1), nn.Conv2d(32, 32, 4, 2, 1),
                                  nn.Conv2d(32, 64, 4, 2, 1), nn.Conv2d(64, 64, 4, 2, 1),
                                  nn.Conv2d(64, 64, 4, 2, 1)])
        self.enc_last = nn.Conv2d(64, 64, 8)
        self.sizes = [s // 64 - 1, s // 32, s // 16 - 1, s // 8, s // 4 - 1, s // 2 - 1]
        self.dec = nn.ModuleList([nn.Conv2d(64, 64, 4, 1, 2) for _ in self.sizes])
        self.out_size = s // 4 - 8
        self.dec7, self.dec8 = nn.Conv2d(64, 64, 3, 1, 1), nn.Conv2d(64, 384, 3, 1, 1)

    def forward(self, x):
        for c in self.enc:
            x = F.relu(c(x))
        x = self.enc_last(x)
        for sz, c in zip(self.sizes, self.dec):
            x = F.relu(c(F.interpolate(x, size=(sz, sz), mode="bilinear")))
        x = F.interpolate(x, size=(self.out_size, self.out_size), mode="bilinear")
        return self.dec8(F.relu(self.dec7(x)))


class EfficientADS(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size
        self.teacher, self.student = PDNSmall(384), PDNSmall(768)
        self.ae = EffADAutoencoder(size)
        self.register_buffer("t_mean", torch.rand(1, 384, 1, 1))
        self.register_buffer("t_std", torch.rand(1, 384, 1, 1) + 0.5)
        self.register_buffer("mean", IMNET_MEAN.clone())
        self.register_buffer("std", IMNET_STD.clone())
        self.qa_st, self.qb_st, self.qa_ae, self.qb_ae = 0.1, 0.9, 0.2, 0.8

    def forward(self, x):
        x = (x - self.mean) / self.std
        t = (self.teacher(x) - self.t_mean) / self.t_std
        s = self.student(x)
        ae = self.ae(x)
        m_st = torch.mean((t - s[:, :384]) ** 2, 1, keepdim=True)
        m_ae = torch.mean((ae - s[:, 384:]) ** 2, 1, keepdim=True)
        m_st = F.interpolate(F.pad(m_st, (4, 4, 4, 4)), size=(self.size, self.size), mode="bilinear")
        m_ae = F.interpolate(F.pad(m_ae, (4, 4, 4, 4)), size=(self.size, self.size), mode="bilinear")
        m_st = 0.1 * (m_st - self.qa_st) / (self.qb_st - self.qa_st)
        m_ae = 0.1 * (m_ae - self.qa_ae) / (self.qb_ae - self.qa_ae)
        return 0.5 * m_st + 0.5 * m_ae


class DinoPatches(nn.Module):
    def __init__(self, width, heads, size):
        super().__init__()
        from transformers import Dinov2Config, Dinov2Model
        # image_size = the crop, so the position table already matches the patch
        # grid and no interpolation is exported. With a pretrained 518 table the
        # interpolation would be a constant TensorRT folds away; same cost.
        cfg = Dinov2Config(image_size=size, patch_size=14, hidden_size=width,
                           num_hidden_layers=12, num_attention_heads=heads, mlp_ratio=4,
                           attn_implementation="eager")
        self.m = Dinov2Model(cfg)

    def forward(self, x):
        return self.m(pixel_values=x).last_hidden_state[:, 1:, :]


class R18L123(nn.Module):
    def __init__(self):
        super().__init__()
        import torchvision.models as M
        m = M.resnet18(weights=None)
        self.stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool)
        self.layer1, self.layer2, self.layer3 = m.layer1, m.layer2, m.layer3

    def forward(self, x):
        f1 = self.layer1(self.stem(x))
        f2 = self.layer2(f1)
        return f1, f2, self.layer3(f2)


def unet_r34():
    import segmentation_models_pytorch as smp
    return smp.Unet(encoder_name="resnet34", encoder_weights=None, classes=3)


CNN_SIZES = [(256, 1), (512, 1), (256, 8)]
VIT_SIZES = [(252, 1), (504, 1), (252, 8)]
MODELS = {
    "wrn50_pc": (lambda s: PatchCoreWRN50(), CNN_SIZES, ["features"]),
    "effad_s":  (lambda s: EfficientADS(s), CNN_SIZES, ["anomaly_map"]),
    "dinov2_s": (lambda s: DinoPatches(384, 6, s), VIT_SIZES, ["patch_tokens"]),
    "dinov2_b": (lambda s: DinoPatches(768, 12, s), VIT_SIZES, ["patch_tokens"]),
    "r18_l123": (lambda s: R18L123(), CNN_SIZES, ["layer1", "layer2", "layer3"]),
    "unet_r34": (lambda s: unet_r34(), CNN_SIZES, ["recon"]),
}


# --------------------------------------------------------------------------
# trtexec parsing (same fields and regexes as scripts/engine_sweep.py)
# --------------------------------------------------------------------------
def mean_of(log, label):
    m = re.search(re.escape(label) + r".*?mean = ([\d.]+) ms", log)
    return float(m.group(1)) if m else float("nan")


def out_shapes(log):
    return re.findall(r"Output binding for \S+ with dimensions ([\dx]+) is created", log)


def nbytes_fp32(dims):
    n = 4
    for d in dims.split("x"):
        n *= int(d)
    return n


def main():
    work, raw_tsv, sum_tsv = sys.argv[1:4]
    only = set(sys.argv[4:])
    os.makedirs(work, exist_ok=True)
    torch.manual_seed(0)

    # 1) export + build, once per engine
    engines = []  # (tag, model, size, batch, params, onnx_path, plan_path)
    for name, (build, sizes, outs) in MODELS.items():
        if only and name not in only:
            continue
        for size, batch in sizes:
            tag = "%s_%d_b%d" % (name, size, batch)
            m = build(size).eval()
            params = sum(p.numel() for p in m.parameters())
            onnx_path = os.path.join(work, tag + ".onnx")
            plan = os.path.join(work, tag + ".plan")
            with torch.no_grad():
                torch.onnx.export(m, (torch.rand(batch, 3, size, size),), onnx_path,
                                  input_names=["images"], output_names=outs,
                                  opset_version=17, dynamo=False)
            r = subprocess.run([TRTEXEC, "--onnx=" + onnx_path, "--fp16", "--saveEngine=" + plan],
                               capture_output=True, text=True)
            log = r.stdout + r.stderr
            if "PASSED" not in log or not os.path.exists(plan):
                tail = [l for l in log.splitlines() if re.search(r"error|Error|failed", l)][-3:]
                print("%-22s BUILD_FAILED %s" % (tag, " | ".join(tail)), flush=True)
                continue
            print("%-22s built  params=%d" % (tag, params), flush=True)
            engines.append((tag, name, size, batch, params, onnx_path, plan))

    # 2) time: REPEATS interleaved rounds, order rotated each round
    raw = open(raw_tsv, "w")
    raw.write("tag\tmodel\tsize\tbatch\trep\tqps\tgpu_ms\th2d_ms\td2h_ms\tlatency_ms\toutput\n")
    rows = {}
    for rep in range(REPEATS):
        k = rep * len(engines) // REPEATS
        for tag, name, size, batch, params, _, plan in engines[k:] + engines[:k]:
            r = subprocess.run([TRTEXEC, "--loadEngine=" + plan], capture_output=True, text=True)
            log = r.stdout + r.stderr
            qm = re.search(r"Throughput: ([\d.]+) qps", log)
            if "PASSED" not in log or not qm:
                print("%-22s rep%d TIMING_FAILED" % (tag, rep + 1), flush=True)
                continue
            v = dict(qps=float(qm.group(1)), gpu=mean_of(log, "GPU Compute Time:"),
                     h2d=mean_of(log, "H2D Latency:"), d2h=mean_of(log, "D2H Latency:"),
                     lat=mean_of(log, "Latency:"), out=";".join(out_shapes(log)))
            rows.setdefault(tag, []).append(v)
            raw.write("%s\t%s\t%d\t%d\t%d\t%.1f\t%.4f\t%.4f\t%.4f\t%.4f\t%s\n"
                      % (tag, name, size, batch, rep + 1, v["qps"], v["gpu"], v["h2d"],
                         v["d2h"], v["lat"], v["out"]))
            raw.flush()
            print("%-22s rep%d  gpu %7.3f ms  h2d %6.3f  d2h %6.3f"
                  % (tag, rep + 1, v["gpu"], v["h2d"], v["d2h"]), flush=True)
    raw.close()

    # 3) medians
    with open(sum_tsv, "w") as f:
        f.write("model\tinput\tbatch\toutput\tin_bytes\tout_bytes\tparams\treps\tqps\tgpu_ms"
                "\th2d_ms\td2h_ms\tnon_engine_ms\ttransport_pct\tgpu_ms_per_frame"
                "\ttotal_ms_per_frame\tgpu_ms_min\tgpu_ms_max\n")
        for tag, name, size, batch, params, _, _ in engines:
            vs = rows.get(tag, [])
            if not vs:
                continue
            med = {k: statistics.median(v[k] for v in vs) for k in ("qps", "gpu", "h2d", "d2h")}
            out = vs[0]["out"]
            ob = sum(nbytes_fp32(d) for d in out.split(";") if d)
            ne = med["h2d"] + med["d2h"]
            f.write("%s\t%dx3x%dx%d\t%d\t%s\t%d\t%d\t%d\t%d\t%.1f\t%.4f\t%.4f\t%.4f\t%.4f\t%.2f"
                    "\t%.4f\t%.4f\t%.4f\t%.4f\n"
                    % (name, batch, size, size, batch, out, batch * 3 * size * size * 4, ob,
                       params, len(vs), med["qps"], med["gpu"], med["h2d"], med["d2h"], ne,
                       100.0 * ne / (med["gpu"] + ne), med["gpu"] / batch,
                       (med["gpu"] + ne) / batch,
                       min(v["gpu"] for v in vs), max(v["gpu"] for v in vs)))


if __name__ == "__main__":
    main()
