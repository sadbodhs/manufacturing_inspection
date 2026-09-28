#!/usr/bin/env python3
"""Phase 1a-2: searching PatchCore's memory bank faster.

Phase 1a put PatchCore's nearest-neighbour search inside the TensorRT engine as
brute force (one matmul + a min) and found the bank, not the backbone, setting
the cost past ~20k patches: 3.6 ms per crop at 100k, 13 ms at 512. This phase
asks what the standard embedding-search tools do with the same job:

  exact (same answer as brute force)
    torch_fp16     brute force in PyTorch, FP16: the in-engine method, outside TRT
    faiss_flat     FAISS GpuIndexFlatL2, FP32
    faiss_flat16   FAISS GpuIndexFlatL2 with FP16 storage and compute
  approximate (may return a different neighbour)
    faiss_ivf      FAISS GpuIndexIVFFlat, nlist ~ 4*sqrt(N), nprobe 1 / 8 / 32
    faiss_ivfpq    FAISS GpuIndexIVFPQ, 64 sub-quantisers x 8 bits, nprobe 8 / 32
    cagra          cuVS CAGRA graph search, graph degree 32, itopk 32 / 64 / 128
                   (only if cuVS installs for this CUDA; its absence is reported)

Banks of 10k and 100k patches; queries of one 256 crop (1,024 patches), one 512
crop (4,096) and eight 256 crops (8,192). Search time only, on the GPU, with
bank and queries already resident (median of 50 timed calls after 10 warm-ups,
CUDA-synchronised). Index build time is reported separately: it is paid once.

RECALL, and why real features. An approximate index can always be made faster
by returning worse neighbours, so its speed means nothing without its recall.
Recall depends on how the embeddings are distributed, and random-weight
features are not distributed like real ones. So this phase, unlike every other,
uses PRETRAINED WideResNet-50-2 (torchvision) features of real images: COCO
val2017 crops, the bank from one set of images and the queries from another.
Two numbers per approximate setting:
  recall@1        fraction of query patches whose returned neighbour is the true one
  score_err_pct   relative error of the image score (max over patches of the
                  nearest-neighbour distance), which is what PatchCore thresholds
This is search fidelity, not defect detection accuracy: nothing here says how
well PatchCore finds defects, only how faithfully each index reproduces exact
search on real features.

PREDICTIONS (written 2026-09-27, before any index was built)

  P1  FAISS flat (exact) beats the in-engine brute force at 100k, because it
      never writes the full distance matrix: >= 2x faster than torch_fp16 at
      100k with 4,096 queries.
  P2  IVF-Flat at nprobe 8 is 5-20x faster than exact at 100k with recall@1
      >= 0.9 and image-score error < 1%.
  P3  IVF-PQ is the fastest FAISS option but its recall@1 falls below 0.8;
      its image-score error stays under 5%, because the image score is a max
      over ~1,000 patches and tolerates individual misses.
  P4  At 10k none of this pays much: exact search is already < 0.5 ms, so an
      index saves less than moving the features out of the engine costs
      (0.24 ms at 256).

Runs in the mi-search container (scripts/phase1a2_search.sh).
Usage: python3 phase1a2_search.py <out_dir>
"""
import glob
import json
import os
import statistics
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

OUT = sys.argv[1]
DEV = "cuda"
torch.manual_seed(0)
np.random.seed(0)


# ---------------- real features: pretrained WRN50 + PatchCore head ----------------
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
        f = torch.cat([f2, f3], 1)                      # B x 1536 x h x w
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
    fs = []
    for i in range(0, len(paths), bs):
        fs.append(feats(load_crops(paths[i:i + bs], size).to(DEV).half()))
    return torch.cat(fs)


# ---------------- timing ----------------
def gpu_time(fn, warm=10, reps=50):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t) * 1000)
    return statistics.median(ts)


def exact_nn(bank16, q16, chunk=4096):
    """Brute force in FP16 (the in-engine method), chunked so memory stays bounded."""
    bsq = (bank16.float() ** 2).sum(1)
    d_out, i_out = [], []
    for s in range(0, q16.shape[0], chunk):
        q = q16[s:s + chunk]
        d2 = (q.float() ** 2).sum(1, keepdim=True) - 2 * (q @ bank16.t()).float() + bsq
        d, i = d2.min(1)
        d_out.append(d.clamp(min=0).sqrt()); i_out.append(i)
    return torch.cat(d_out), torch.cat(i_out)


def image_scores(dist, per_image):
    return dist.view(-1, per_image).max(1).values


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    feats = feature_extractor()
    imgs = sorted(glob.glob("/coco/val2017/*.jpg"))
    rng = np.random.default_rng(0)
    rng.shuffle(imgs)
    bank_imgs, query_imgs = imgs[:110], imgs[200:216]

    # bank: 110 images x 1,024 patches at 256 = 112,640 real patches; subsample to 10k / 100k
    bank_all = patches(feats, bank_imgs, 256)
    queries = {"256x1": (patches(feats, query_imgs[:1], 256), 1024),
               "512x1": (patches(feats, query_imgs[:1], 512), 4096),
               "256x8": (patches(feats, query_imgs[:8], 256), 1024)}
    # score_err is judged over 16 query images at 256, so it is not one image's luck
    q_eval = patches(feats, query_imgs, 256)

    import faiss
    import faiss.contrib.torch_utils  # noqa: F401  (search takes GPU torch tensors: no host copies)
    res = faiss.StandardGpuResources()
    try:
        from cuvs.neighbors import cagra
        import cupy as cp
        have_cagra = True
    except Exception as e:  # its absence is a result
        have_cagra = False
        rows.append(dict(method="cagra", note="not available: %s" % str(e)[:120]))

    for n in (10_000, 100_000):
        idx = torch.randperm(bank_all.shape[0], device=DEV)[:n]
        bank = bank_all[idx].contiguous()
        bank16 = bank.half()
        # Reference = FP32 exact search (FAISS flat). The first run used the FP16
        # brute force as the reference and found it disagrees with FP32 on 1-2% of
        # nearest neighbours (near-ties flipped by FP16 rounding), so FP16 is itself
        # an approximation here and is scored like one. First run: *_run1.tsv.
        ref_ix = faiss.GpuIndexFlatL2(res, 1536)
        ref_ix.add(bank)
        rd, ri = ref_ix.search(q_eval, 1)
        ev_d, ev_i = rd.view(-1).clamp(min=0).sqrt(), ri.view(-1)
        ev_score = image_scores(ev_d, 1024)

        def fidelity(d, i):
            d = torch.as_tensor(d, device=DEV).float().view(-1).clamp(min=0).sqrt()
            i = torch.as_tensor(i, device=DEV).view(-1)
            rec = (i == ev_i).float().mean().item()
            err = ((image_scores(d, 1024) - ev_score).abs() / ev_score).mean().item() * 100
            return rec, err

        # brute force in PyTorch FP16 (the in-engine method, unfused), scored against FP32
        td, ti = exact_nn(bank16, q_eval.half())
        rec16, err16 = fidelity(td ** 2, ti)
        for qn, (q, _) in queries.items():
            q16 = q.half()
            rows.append(dict(method="torch_fp16", bank=n, queries=qn, nq=q.shape[0], param="",
                             ms=gpu_time(lambda: exact_nn(bank16, q16)), recall1=rec16, score_err_pct=err16))

        # exact: FAISS flat, FP32 and FP16
        for name, fp16 in (("faiss_flat", False), ("faiss_flat16", True)):
            cfg = faiss.GpuIndexFlatConfig()
            cfg.useFloat16 = fp16
            index = faiss.GpuIndexFlatL2(res, 1536, cfg)
            t = time.perf_counter(); index.add(bank); build = (time.perf_counter() - t) * 1000
            d, i = index.search(q_eval, 1)
            rec, err = fidelity(d, i)
            for qn, (q, _) in queries.items():
                q = q.contiguous()
                rows.append(dict(method=name, bank=n, queries=qn, nq=q.shape[0], param="", build_ms=build,
                                 ms=gpu_time(lambda: index.search(q, 1)), recall1=rec, score_err_pct=err))

        # approximate: IVF-Flat and IVF-PQ
        nlist = int(4 * np.sqrt(n))
        for name, make, probes in (
                ("faiss_ivf", lambda: faiss.GpuIndexIVFFlat(res, 1536, nlist, faiss.METRIC_L2), (1, 8, 32)),
                ("faiss_ivfpq", lambda: _ivfpq(faiss, res, nlist), (8, 32))):
            index = make()
            t = time.perf_counter(); index.train(bank); index.add(bank); build = (time.perf_counter() - t) * 1000
            for nprobe in probes:
                index.nprobe = nprobe
                d, i = index.search(q_eval, 1)
                rec, err = fidelity(d, i)
                for qn, (q, _) in queries.items():
                    q = q.contiguous()
                    rows.append(dict(method=name, bank=n, queries=qn, nq=q.shape[0],
                                     param="nlist=%d nprobe=%d" % (nlist, nprobe), build_ms=build,
                                     ms=gpu_time(lambda: index.search(q, 1)), recall1=rec, score_err_pct=err))

        # approximate: CAGRA
        if have_cagra:
            b_cp = cp.asarray(bank)
            t = time.perf_counter()
            gidx = cagra.build(cagra.IndexParams(graph_degree=32, intermediate_graph_degree=64), b_cp)
            cp.cuda.runtime.deviceSynchronize(); build = (time.perf_counter() - t) * 1000
            for itopk in (32, 64, 128):
                sp = cagra.SearchParams(itopk_size=itopk)
                d, i = cagra.search(sp, gidx, cp.asarray(q_eval), 1)
                rec, err = fidelity(torch.as_tensor(cp.asarray(d), device=DEV), torch.as_tensor(cp.asarray(i).astype(cp.int64), device=DEV))
                for qn, (q, _) in queries.items():
                    qc = cp.asarray(q)
                    rows.append(dict(method="cagra", bank=n, queries=qn, nq=q.shape[0],
                                     param="degree=32 itopk=%d" % itopk, build_ms=build,
                                     ms=gpu_time(lambda: cagra.search(sp, gidx, qc, 1)), recall1=rec,
                                     score_err_pct=err))
        print("bank %d done" % n, flush=True)

    cols = ["method", "bank", "queries", "nq", "param", "ms", "recall1", "score_err_pct", "build_ms", "note"]
    with open(os.path.join(OUT, "phase1a2_search.tsv"), "w") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join("%.4f" % r[c] if isinstance(r.get(c), float) else str(r.get(c, "")) for c in cols) + "\n")
    with open(os.path.join(OUT, "phase1a2_env.json"), "w") as f:
        json.dump(dict(faiss=faiss.__version__, torch=torch.__version__, cagra=have_cagra,
                       bank_patches=int(bank_all.shape[0]), bank_images=len(bank_imgs)), f)
    print("wrote %d rows" % len(rows), flush=True)


def _ivfpq(faiss, res, nlist):
    cfg = faiss.GpuIndexIVFPQConfig()
    cfg.useFloat16LookupTables = True   # needed for 64 sub-quantisers on the GPU
    return faiss.GpuIndexIVFPQ(res, 1536, nlist, 64, 8, faiss.METRIC_L2, cfg)


if __name__ == "__main__":
    main()
