#!/usr/bin/env python3
"""Phase 5 prep: real features, exact references and TensorRT engines.

Builds everything the footprint workers load, once, so no worker pays for it:

  bank_1m_f32.npy   1,000,000 pretrained WRN50 patch features of COCO val2017
                    crops at 256 (1,000 images x 1,024 patches, shuffled, cut
                    to 1M). Banks of 10k and 100k are nested prefixes of a fixed
                    permutation (perm.npy), sorted, so every method sees the same
                    rows in the same order.
  q256 / q512       the query: one crop at 256 (1,024 patches) and at 512 (4,096)
  qeval             16 crops at 256 (16,384 patches), for recall
  ref_<n>_{d,i}     FP32 exact nearest neighbours of qeval (FAISS flat on GPU),
                    the reference every method's recall is scored against
  trt_<n>_<N>.plan  TensorRT brute-force search engines: the in-engine method of
                    Phase 1a with the backbone removed. The bank is a graph
                    constant (FP32 weights, FP16 build flag, as in Phase 1a);
                    output is the top-1 (distance, index) per query patch.

Query images are the same 16 COCO images as Phase 1a-2 (shuffled indices
200-215); bank images are the first 1,000 others.

Usage (inside mi-search): python3 phase5_prep.py
"""
import glob
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

D = "/mi/data/search"
DEV = "cuda"
BANKS = (10_000, 100_000, 1_000_000)
QN = (1024, 4096)


def feature_extractor():
    import torchvision.models as M
    m = M.wide_resnet50_2(weights=M.Wide_ResNet50_2_Weights.IMAGENET1K_V1).eval().to(DEV).half()
    pool = torch.nn.AvgPool2d(3, 1, 1)

    @torch.no_grad()
    def feats(x):
        x = m.maxpool(m.relu(m.bn1(m.conv1(x))))
        f2 = m.layer2(m.layer1(x))
        f3 = m.layer3(f2)
        f2, f3 = pool(f2), pool(f3)
        f3 = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        f = torch.cat([f2, f3], 1)
        return f.flatten(2).transpose(1, 2).reshape(-1, 1536).float()
    return feats


def load_crops(paths, size):
    from PIL import Image
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    out = []
    for p in paths:
        im = Image.open(p).convert("RGB")
        s = min(im.size)
        im = im.crop(((im.width - s) // 2, (im.height - s) // 2, (im.width + s) // 2, (im.height + s) // 2))
        t = torch.from_numpy(np.asarray(im.resize((size, size)), dtype=np.float32) / 255.0).permute(2, 0, 1)
        out.append((t - mean) / std)
    return torch.stack(out)


def patches(feats, paths, size, bs=16):
    per = (size // 8) ** 2
    out = np.empty((len(paths) * per, 1536), dtype=np.float32)   # preallocated: 1,000 images is 6.3 GB
    for i in range(0, len(paths), bs):
        f = feats(load_crops(paths[i:i + bs], size).to(DEV).half()).cpu().numpy()
        out[i * per:i * per + f.shape[0]] = f
    return out


def build_trt(bank, n_q, path):
    import tensorrt as trt
    log = trt.Logger(trt.Logger.WARNING)
    b = trt.Builder(log)
    net = b.create_network(0)
    q = net.add_input("q", trt.float32, (n_q, 1536))
    bt = np.ascontiguousarray(bank.T)                       # 1536 x M, FP32 (FP16 flag converts)
    bsq = np.ascontiguousarray((bank.astype(np.float64) ** 2).sum(1).astype(np.float32).reshape(1, -1))
    B = net.add_constant(bt.shape, trt.Weights(bt)).get_output(0)
    S = net.add_constant(bsq.shape, trt.Weights(bsq)).get_output(0)
    two = np.array([[-2.0]], dtype=np.float32)
    T = net.add_constant((1, 1), trt.Weights(two)).get_output(0)
    mm = net.add_matrix_multiply(q, trt.MatrixOperation.NONE, B, trt.MatrixOperation.NONE).get_output(0)
    qq = net.add_elementwise(q, q, trt.ElementWiseOperation.PROD).get_output(0)
    qs = net.add_reduce(qq, trt.ReduceOperation.SUM, 1 << 1, True).get_output(0)
    x = net.add_elementwise(mm, T, trt.ElementWiseOperation.PROD).get_output(0)
    x = net.add_elementwise(x, qs, trt.ElementWiseOperation.SUM).get_output(0)
    x = net.add_elementwise(x, S, trt.ElementWiseOperation.SUM).get_output(0)
    tk = net.add_topk(x, trt.TopKOperation.MIN, 1, 1 << 1)
    tk.get_output(0).name, tk.get_output(1).name = "d", "i"
    net.mark_output(tk.get_output(0)); net.mark_output(tk.get_output(1))
    cfg = b.create_builder_config()
    cfg.set_flag(trt.BuilderFlag.FP16)
    t = time.perf_counter()
    ser = b.build_serialized_network(net, cfg)
    ms = (time.perf_counter() - t) * 1000
    if ser is None:
        return dict(ok=False, build_ms=ms)
    with open(path, "wb") as f:
        f.write(ser)
    return dict(ok=True, build_ms=ms, plan_mb=os.path.getsize(path) / 2**20)


def main():
    os.makedirs(D, exist_ok=True)
    info = {}
    if not os.path.exists(f"{D}/bank_1m_f32.npy"):
        feats = feature_extractor()
        imgs = sorted(glob.glob("/coco/val2017/*.jpg"))
        rng = np.random.default_rng(0)
        rng.shuffle(imgs)
        qimgs = imgs[200:216]
        bimgs = [p for k, p in enumerate(imgs) if not 200 <= k < 216][:1000]
        np.save(f"{D}/q256.npy", patches(feats, qimgs[:1], 256))
        np.save(f"{D}/q512.npy", patches(feats, qimgs[:1], 512))
        np.save(f"{D}/qeval.npy", patches(feats, qimgs, 256))
        bank = patches(feats, bimgs, 256)
        sh = np.random.default_rng(1).permutation(bank.shape[0])[:1_000_000]
        np.save(f"{D}/bank_1m_f32.npy", np.ascontiguousarray(bank[sh]))
        np.save(f"{D}/perm.npy", np.random.default_rng(2).permutation(1_000_000))
        del bank, feats
        torch.cuda.empty_cache()
        print("features done", flush=True)

    import faiss
    full = np.load(f"{D}/bank_1m_f32.npy", mmap_mode="r")
    perm = np.load(f"{D}/perm.npy")
    qeval = np.load(f"{D}/qeval.npy")
    res = faiss.StandardGpuResources()
    for n in BANKS:
        sel = np.sort(perm[:n])
        bank = np.ascontiguousarray(full[sel])
        if not os.path.exists(f"{D}/ref_{n}_i.npy"):
            ix = faiss.GpuIndexFlatL2(res, 1536)
            ix.add(bank)
            d, i = ix.search(qeval, 1)
            np.save(f"{D}/ref_{n}_d.npy", d.ravel()); np.save(f"{D}/ref_{n}_i.npy", i.ravel())
            del ix
        for nq in QN:
            p = f"{D}/trt_{n}_{nq}.plan"
            if not os.path.exists(p):
                info[f"trt_{n}_{nq}"] = r = build_trt(bank, nq, p)
                print(f"trt {n} {nq}", r, flush=True)
        print("bank %d done" % n, flush=True)
    old = json.load(open(f"{D}/prep.json")) if os.path.exists(f"{D}/prep.json") else {}
    old.update(info)
    json.dump(old, open(f"{D}/prep.json", "w"), indent=1)


if __name__ == "__main__":
    main()
