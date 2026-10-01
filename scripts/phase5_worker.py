#!/usr/bin/env python3
"""Phase 5 worker: one search method, one bank, in a fresh process.

A fresh process per (method, bank) so the method's fixed cost (CUDA context,
library scratch memory) is measured from zero, as a deployment would pay it.
The worker does the work and prints a JSON line at each stage boundary; GPU
memory and utilisation are sampled from OUTSIDE by phase5_sampler.py (NVML,
device-wide; the driver checks nothing else is on the GPU first), so the
numbers include every allocator's cache, exactly what the card has to hold.

Stages: start -> host_loaded -> ctx (CUDA context) -> lib (library ready, no
data) -> built (bank on GPU / index built) -> per query shape: latency, then a
5 s back-to-back window (max crops/s, utilisation, CPU cores) -> rate sweep at
10/25/50% of max (rep 1 only) -> recall on 16 crops.

Methods
  trt_bf / trt_bf512   TensorRT brute force, bank as engine constant (Phase 1a's
                       in-engine search), engine for a 256 / 512 crop
  faiss_flat16         FAISS GpuIndexFlatL2, FP16 storage (exact)
  faiss_ivf            FAISS GpuIndexIVFFlat, nlist 4*sqrt(N), nprobe 8 / 32
  faiss_ivfpq          FAISS GpuIndexIVFPQ, 64 x 8 bit, nprobe 8 / 32
  cagra                cuVS CAGRA, graph degree 32, itopk 64 / 128
  cpu_flat             FAISS IndexFlatL2 on 8 CPU threads (exact), no GPU
  cpu_ivf              FAISS IndexIVFFlat on 8 CPU threads, nprobe 32, no GPU

PREDICTIONS (written 2026-10-01, before any worker ran; prep had built only
features, references and TensorRT engines)

  F1  Fixed cost. Every GPU method pays >= 250 MB before any bank (CUDA context).
      FAISS methods pay >= 1 GB more (its scratch reservation), so at a 100k
      bank FAISS flat16's fixed cost exceeds its own bank (~300 MB).
  F2  Bank storage at 1M (resident minus fixed cost): IVF-PQ < 0.25 GB;
      TensorRT and FAISS flat16 2.9-3.5 GB (FP16); IVF-Flat and CAGRA >= 5.8 GB
      (they keep FP32 vectors). Only PQ is lighter than exact search.
  F3  Transient. TensorRT brute force at a 512 crop needs at least the
      4,096 x bank FP16 distance matrix: >= 0.8 GB at 100k, >= 8 GB at 1M, the
      largest peak of any method at 1M (or it does not fit).
  F4  Utilisation. nvidia-smi reads >= 90% for every GPU method running back to
      back, and in the rate sweep tracks duty (rate x latency) within 15 points:
      it measures time busy, so it cannot tell a light kernel from a heavy one.
  F5  CPU. cpu_flat at 100k is >= 20x slower than faiss_flat16 per 256 crop and
      uses >= 6 of its 8 threads' cores at max rate; cpu_ivf (nprobe 32) at 100k
      takes < 50 ms per 256 crop.
  F6  Continuity. GPU latencies at 100k are within 25% of Phase 1a-2 for the
      same settings, and TensorRT's search-only engine at 256 within 25% of the
      2.96 ms in-engine search.
  F7  Scaling 100k -> 1M. Exact methods slow >= 8x; IVF at fixed nprobe slows
      <= 4x (nlist grows sqrt(10), so each probed list grows ~3.2x).

Usage: python3 phase5_worker.py <method> <bank> <rep>
"""
import json
import os
import statistics
import sys
import time

import numpy as np

D = "/mi/data/search"
METHOD, N, REP = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
GPU = not METHOD.startswith("cpu_")
THREADS = 8


def mark(stage, **kw):
    print(json.dumps(dict(stage=stage, t=time.monotonic(), **kw)), flush=True)


def rss_mb():
    for line in open("/proc/self/status"):
        if line.startswith("VmRSS"):
            return int(line.split()[1]) / 1024
    return 0.0


mark("start")
full = np.load(f"{D}/bank_1m_f32.npy", mmap_mode="r")
sel = np.sort(np.load(f"{D}/perm.npy")[:N])
bank = np.ascontiguousarray(full[sel]) if N < full.shape[0] else np.ascontiguousarray(full)
Q = {"256": np.load(f"{D}/q256.npy"), "512": np.load(f"{D}/q512.npy")}
QEVAL = np.load(f"{D}/qeval.npy")
REF_I = np.load(f"{D}/ref_{N}_i.npy")
REF_D = np.sqrt(np.maximum(np.load(f"{D}/ref_{N}_d.npy"), 0))
REF_S = REF_D.reshape(-1, 1024).max(1)
mark("host_loaded")

if GPU:
    import torch
    torch.zeros(1, device="cuda")
    torch.cuda.synchronize()
    sync = torch.cuda.synchronize
else:
    sync = lambda: None  # noqa: E731
mark("ctx")


# ---------------- methods: each returns (params, set_param, search(shape), eval(q)) ----------------
def setup_trt():
    import tensorrt as trt
    rt = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    mark("lib")
    nq = 4096 if METHOD == "trt_bf512" else 1024
    t = time.perf_counter()
    eng = rt.deserialize_cuda_engine(open(f"{D}/trt_{N}_{nq}.plan", "rb").read())
    ctx = eng.create_execution_context()
    qb = torch.empty(nq, 1536, device="cuda")
    od = torch.empty(nq, 1, device="cuda")
    oi = torch.empty(nq, 1, dtype=torch.int32, device="cuda")
    for name, buf in (("q", qb), ("d", od), ("i", oi)):
        ctx.set_tensor_address(name, buf.data_ptr())
    sync()
    build = (time.perf_counter() - t) * 1000
    stream = torch.cuda.current_stream().cuda_stream
    shape = "512" if nq == 4096 else "256"

    def run():
        ctx.execute_async_v3(stream)

    def search(s):
        qb.copy_(torch.from_numpy(Q[s]).cuda())
        return run

    def ev(_):
        ds, is_ = [], []
        q = torch.from_numpy(QEVAL).cuda()
        for k in range(0, q.shape[0], nq):
            qb.copy_(q[k:k + nq]); run(); sync()
            ds.append(od.view(-1).float().cpu().numpy()); is_.append(oi.view(-1).cpu().numpy())
        return np.maximum(np.concatenate(ds), 0), np.concatenate(is_)
    return [""], lambda p: None, search, ev, [shape], build, dict(device_mem_mb=eng.device_memory_size_v2 / 2**20)


def setup_faiss():
    import faiss
    import faiss.contrib.torch_utils  # noqa: F401
    res = faiss.StandardGpuResources()
    mark("lib")
    t = time.perf_counter()
    nlist = int(4 * np.sqrt(N))
    if METHOD == "faiss_flat16":
        cfg = faiss.GpuIndexFlatConfig(); cfg.useFloat16 = True
        ix = faiss.GpuIndexFlatL2(res, 1536, cfg); params = [""]
    elif METHOD == "faiss_ivf":
        ix = faiss.GpuIndexIVFFlat(res, 1536, nlist, faiss.METRIC_L2); params = [8, 32]
    else:
        cfg = faiss.GpuIndexIVFPQConfig(); cfg.useFloat16LookupTables = True
        ix = faiss.GpuIndexIVFPQ(res, 1536, nlist, 64, 8, faiss.METRIC_L2, cfg); params = [8, 32]
    if METHOD != "faiss_flat16":
        ix.train(bank)
    ix.add(bank)
    sync()
    build = (time.perf_counter() - t) * 1000
    qs = {s: torch.from_numpy(Q[s]).cuda() for s in Q}

    def setp(p):
        if p != "":
            ix.nprobe = p

    def search(s):
        q = qs[s]
        return lambda: ix.search(q, 1)

    def ev(_):
        d, i = ix.search(torch.from_numpy(QEVAL).cuda(), 1)
        return d.view(-1).clamp(min=0).cpu().numpy(), i.view(-1).cpu().numpy()
    return params, setp, search, ev, ["256", "512"], build, dict(nlist=nlist if METHOD != "faiss_flat16" else 0)


def setup_cagra():
    import cupy as cp
    from cuvs.neighbors import cagra
    cp.zeros(1); cp.cuda.runtime.deviceSynchronize()
    mark("lib")
    t = time.perf_counter()
    b = cp.asarray(bank)
    idx = cagra.build(cagra.IndexParams(graph_degree=32, intermediate_graph_degree=64), b)
    cp.cuda.runtime.deviceSynchronize()
    build = (time.perf_counter() - t) * 1000
    qs = {s: cp.asarray(Q[s]) for s in Q}
    sp = {}

    def setp(p):
        sp["p"] = cagra.SearchParams(itopk_size=p)

    def search(s):
        q = qs[s]
        return lambda: cagra.search(sp["p"], idx, q, 1)

    def ev(_):
        d, i = cagra.search(sp["p"], idx, cp.asarray(QEVAL), 1)
        return cp.asnumpy(cp.asarray(d)).ravel().clip(0), cp.asnumpy(cp.asarray(i)).ravel().astype(np.int64)
    return [64, 128], setp, search, ev, ["256", "512"], build, {}


def setup_cpu():
    import faiss
    faiss.omp_set_num_threads(THREADS)
    mark("lib")
    t = time.perf_counter()
    if METHOD == "cpu_flat":
        ix = faiss.IndexFlatL2(1536); params = [""]
    else:
        nlist = int(4 * np.sqrt(N))
        quant = faiss.IndexFlatL2(1536)
        ix = faiss.IndexIVFFlat(quant, 1536, nlist); params = [32]
        ix.train(bank)
    ix.add(bank)
    build = (time.perf_counter() - t) * 1000

    def setp(p):
        if p != "":
            ix.nprobe = p

    def search(s):
        q = Q[s]
        return lambda: ix.search(q, 1)

    def ev(_):
        d, i = ix.search(QEVAL, 1)
        return np.maximum(d.ravel(), 0), i.ravel()
    return params, setp, search, ev, ["256", "512"], build, dict(threads=THREADS)


SETUP = dict(trt_bf=setup_trt, trt_bf512=setup_trt, faiss_flat16=setup_faiss, faiss_ivf=setup_faiss,
             faiss_ivfpq=setup_faiss, cagra=setup_cagra, cpu_flat=setup_cpu, cpu_ivf=setup_cpu)
params, setp, search, ev, shapes, build_ms, extra = SETUP[METHOD]()
mark("built", build_ms=build_ms, rss_mb=rss_mb(), **extra)
del bank  # a deployment keeps only the index's copy (GPU memory, or host memory for cpu_*)
import gc  # noqa: E402
gc.collect()
mark("host_freed", rss_mb=rss_mb())


def cpu_s():
    t = os.times()
    return t.user + t.system


def latency(fn, warm=5, reps=50, budget=8.0):
    for _ in range(warm):
        fn(); sync()
    ts, t0 = [], time.perf_counter()
    while len(ts) < reps and (len(ts) < 5 or time.perf_counter() - t0 < budget):
        t = time.perf_counter(); fn(); sync(); ts.append((time.perf_counter() - t) * 1000)
    return statistics.median(ts), len(ts)


def window(fn, secs, period=None):
    """Back-to-back (period None) or paced calls for `secs`; returns calls/s and CPU cores used."""
    n, c0, t0 = 0, cpu_s(), time.perf_counter()
    due = t0
    while time.perf_counter() - t0 < secs:
        if period:
            now = time.perf_counter()
            if now < due:
                time.sleep(due - now)
            due += period
        fn(); sync(); n += 1
    wall = time.perf_counter() - t0
    return n / wall, (cpu_s() - c0) / wall


for p in params:
    setp(p)
    for s in shapes:
        fn = search(s)
        tag = "%s:%s" % (p, s)
        mark("lat_begin", tag=tag)
        ms, calls = latency(fn)
        mark("lat_end", tag=tag, ms=ms, calls=calls)
        mark("max_begin", tag=tag)
        rate, cores = window(fn, 5.0 if ms < 500 else 10.0)
        mark("max_end", tag=tag, rate=rate, cores=cores)
        if REP == 1 and GPU and s == "256" and p == params[-1]:
            for frac in (0.1, 0.25, 0.5):
                mark("sweep_begin", tag=tag, frac=frac)
                r, c = window(fn, 4.0, period=1.0 / (frac * rate))
                mark("sweep_end", tag=tag, frac=frac, rate=r, cores=c)
    d2, ix = ev(p)
    d = np.sqrt(d2)
    rec = float((ix == REF_I).mean())
    err = float((np.abs(d.reshape(-1, 1024).max(1) - REF_S) / REF_S).mean() * 100)
    mark("recall", param=p, recall1=rec, score_err_pct=err)
mark("done", rss_mb=rss_mb())
