#!/usr/bin/env python3
"""Every figure on the site, from the committed results.

  encoders.png          cost per crop (compute + transport) and batch 1 vs 8
  big-models.png        per-image change from batching, five models
  patchcore-search.png  cost per crop against bank size, with the backbone-only line
  embedding-search.png  search time against recall, TensorRT brute force as reference
  stage3.png            Grounding DINO cost by phrases, text live vs cached
  line-a.png            (if present) fast-path p99 against cameras, per K
  line-b.png            (if present) fast-path p99 and stage-3 skips against flag rate

Colours: two validated series colours (blue, orange) and grey for references.
Usage: python3 scripts/plot_all.py [repo_root]   (RESULTS_DIR overrides <root>/results)
"""
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.ticker  # noqa: E402
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = sys.argv[1] if len(sys.argv) > 1 else "."
RES = os.environ.get("RESULTS_DIR", os.path.join(ROOT, "results"))
IMG = os.path.join(ROOT, "docs", "img")
os.makedirs(IMG, exist_ok=True)

BLUE, ACCENT, GREY, INK, MUTED, GRID = "#1a73e8", "#e8710a", "#9aa0a6", "#202124", "#5f6368", "#e8eaed"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "axes.edgecolor": MUTED, "xtick.color": INK, "ytick.color": INK})


def tsv(name):
    with open(os.path.join(RES, name), newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def grid(ax, axis="x"):
    ax.grid(axis=axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, name), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", name)


def f(r, k):
    return float(r[k])


# ------------------------------------------------------------------ encoders
def encoders():
    rows = tsv("inspection_encoders.tsv")
    small = {r["model"]: r for r in rows if r["batch"] == "1" and r["input"].endswith(("x256x256", "x252x252"))}
    b8 = {r["model"]: r for r in rows if r["batch"] == "8"}
    y8 = [r for r in tsv("phase0_batching.tsv") if r["model"] == "yolov8s" and r["batch"] == "1"][0]
    label = {"r18_l123": "ResNet-18 L1-3", "dinov2_s": "DINOv2 ViT-S", "unet_r34": "U-Net R34 (reference)",
             "wrn50_pc": "WRN50 + PatchCore head", "effad_s": "EfficientAD-S", "dinov2_b": "DINOv2 ViT-B"}
    order = sorted(small, key=lambda m: f(small[m], "total_ms_per_frame"))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1.15, 1]})
    names = [label[m] for m in order] + ["yolov8s at 640 (stage 1)"]
    gpu = [f(small[m], "gpu_ms") for m in order] + [f(y8, "gpu_ms")]
    tr = [f(small[m], "non_engine_ms") for m in order] + [f(y8, "non_engine_ms")]
    ys = list(range(len(names)))[::-1]
    for i, (yy, g, t) in enumerate(zip(ys, gpu, tr)):
        ref = i == len(names) - 1
        a1.barh(yy, g, height=0.62, color=GREY if ref else BLUE)
        a1.barh(yy, t - 0.012, left=g + 0.012, height=0.62, color=GREY if ref else ACCENT, alpha=0.55 if ref else 1)
        a1.text(g + t + 0.03, yy, "%.2f ms" % (g + t), va="center", fontsize=10, color=INK)
    a1.axvline(gpu[-1] + tr[-1], color=GREY, linestyle=":", linewidth=1.2)
    a1.set_yticks(ys); a1.set_yticklabels(names)
    a1.set_xlim(0, 1.7); a1.set_xlabel("ms per crop (crop 256, ViTs 252; batch 1)")
    a1.set_title("Every encoder costs about as much as finding the part, or less", loc="left", fontsize=11.5)
    a1.legend(handles=[Patch(color=BLUE, label="GPU compute"), Patch(color=ACCENT, label="transport (H2D + D2H)"),
                       Patch(color=GREY, label="stage 1, for scale")],
              loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, frameon=False, fontsize=10)
    grid(a1)
    order2 = sorted(b8, key=lambda m: f(small[m], "gpu_ms"))
    for yy, m in zip(list(range(len(order2)))[::-1], order2):
        g1, g8 = f(small[m], "gpu_ms"), f(b8[m], "gpu_ms_per_frame")
        a2.plot([g8, g1], [yy, yy], color="#dadce0", linewidth=2, zorder=1)
        a2.scatter([g1], [yy], s=64, color=BLUE, zorder=3, edgecolor="white", linewidth=2)
        a2.scatter([g8], [yy], s=64, color=ACCENT, zorder=3, edgecolor="white", linewidth=2)
        a2.text(g1 + 0.04, yy, "%+.0f%%" % (100 * (g8 / g1 - 1)), va="center", fontsize=10, color=INK)
    a2.set_yticks(list(range(len(order2)))[::-1]); a2.set_yticklabels([label[m] for m in order2])
    a2.set_xlim(0, 1.55); a2.set_xlabel("GPU ms per crop")
    a2.legend(handles=[Line2D([], [], marker="o", ls="", ms=8, color=BLUE, label="batch 1"),
                       Line2D([], [], marker="o", ls="", ms=8, color=ACCENT, label="batch 8, per crop")],
              loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, frameon=False, fontsize=10)
    a2.set_title("Batching 8 crops saves 45-62% for most (yolov8s frames: -39%)", loc="left", fontsize=11.5)
    grid(a2)
    save(fig, "encoders.png")


# ------------------------------------------------------------------ big models
def big_models():
    rows = tsv("phase0_batching.tsv")
    label = {"yolov8s": "yolov8s, 640", "rtdetr_l": "RT-DETR-L, 640", "dinov2_l": "DINOv2 ViT-L, 518",
             "sam_h_enc": "SAM ViT-H encoder, 1024", "sam_b_enc": "SAM ViT-B encoder, 1024"}
    b1 = {r["model"]: f(r, "gpu_ms_per_frame") for r in rows if r["batch"] == "1"}
    b8 = {r["model"]: f(r, "gpu_ms_per_frame") for r in rows if r["batch"] != "1"}
    order = ["yolov8s", "rtdetr_l", "dinov2_l", "sam_h_enc", "sam_b_enc"]
    ch = [100 * (b8[m] / b1[m] - 1) for m in order]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ys = list(range(len(order)))[::-1]
    for yy, m, c in zip(ys, order, ch):
        ax.barh(yy, c, height=0.6, color=BLUE if c < 0 else ACCENT)
        ax.text(1.5 if c < 0 else c + 1.5, yy, "%+.1f%%   (%.2f -> %.2f ms)" % (c, b1[m], b8[m]),
                va="center", ha="left", fontsize=10, color=INK)
    ax.axvline(0, color=MUTED, linewidth=1)
    ax.set_yticks(ys); ax.set_yticklabels([label[m] for m in order])
    ax.set_xlim(-45, 60)
    ax.set_xlabel("change in GPU time per image, batch 8 vs batch 1")
    ax.set_title("Batching helps by architecture, not by size: SAM-B gets slower", loc="left", fontsize=11.5)
    grid(ax)
    save(fig, "big-models.png")


# ------------------------------------------------------------------ patchcore search
def patchcore_search():
    rows = tsv("phase1a_patchcore.tsv")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=False)
    for ax, (shape, title) in zip(axes, (("1x3x256x256", "256 crop, batch 1"), ("1x3x512x512", "512 crop, batch 1"),
                                         ("8x3x256x256", "256 crop, batch 8 (per crop)"))):
        rs = [r for r in rows if r["input"] == shape]
        bb = [f(r, "total_ms_per_frame") for r in rs if r["model"] == "wrn50_pc_backbone"][0]
        pts = sorted((int(r["model"].split("search")[1].rstrip("k")) * 1000, f(r, "total_ms_per_frame"))
                     for r in rs if "search" in r["model"])
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=BLUE, marker="o", ms=7, lw=2,
                markeredgecolor="white", markeredgewidth=2)
        for x, yv in pts:
            ax.annotate("%.2f" % yv, (x, yv), textcoords="offset points", xytext=(0, 8), ha="center",
                        fontsize=9.5, color=INK)
        ax.axhline(bb, color=GREY, ls="--", lw=1.5)
        ax.text(0.03, 0.95, "dashed: backbone only,\nfeatures shipped out (%.2f ms)" % bb, transform=ax.transAxes,
                va="top", fontsize=9, color=MUTED)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: "%g" % v))
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xticks([1000, 10000, 100000]); ax.set_xticklabels(["1k", "10k", "100k"])
        ax.set_xlabel("memory bank (patches)")
        ax.set_title(title, loc="left", fontsize=11)
        grid(ax, "y")
    axes[0].set_ylabel("ms per crop, total (log)")
    fig.suptitle("Searching inside the engine is cheaper than not searching, up to ~8k patches",
                 x=0.01, ha="left", fontsize=12)
    save(fig, "patchcore-search.png")


# ------------------------------------------------------------------ embedding search
def embedding_search():
    rows = [r for r in tsv("phase1a2_search.tsv") if r.get("bank") == "100000" and r.get("queries") == "256x1"]
    short = {"torch_fp16": "PyTorch FP16", "faiss_flat": "FAISS flat", "faiss_flat16": "FAISS flat FP16",
             "faiss_ivf": "IVF", "faiss_ivfpq": "IVF-PQ", "cagra": "CAGRA"}
    fig, ax = plt.subplots(figsize=(10, 5))
    for r in rows:
        x, yv = f(r, "recall1"), f(r, "ms")
        ax.scatter([x], [yv], s=60, color=BLUE, edgecolor="white", linewidth=2, zorder=3)
        p = r["param"].replace("nlist=1264 ", "").replace("degree=32 ", "")
        lab = ("%s %s" % (short[r["method"]], p)).strip()
        off, ha = {"FAISS flat FP16": ((-8, 6), "right"), "FAISS flat": ((8, -4), "left"),
                   "PyTorch FP16": ((8, -2), "left"), "CAGRA itopk=128": ((-8, -12), "right")}.get(lab, ((6, 4), "left"))
        ax.annotate(lab, (x, yv), textcoords="offset points", xytext=off, ha=ha, fontsize=9, color=INK)
    ax.axhline(2.96, color=ACCENT, lw=2)
    ax.text(0.2, 2.96, "TensorRT in-engine brute force, 2.96 ms", va="bottom", fontsize=10, color=INK)
    ax.set_yscale("log"); ax.set_xlim(0.18, 1.12)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: "%g" % v))
    ax.set_xlabel("recall@1 against FP32 exact search (real WRN50 features)")
    ax.set_ylabel("search time, ms (log)")
    ax.set_title("100k-patch bank, one 256 crop: nothing with recall >= 0.9 comes near brute force",
                 loc="left", fontsize=11.5)
    grid(ax, "both")
    save(fig, "embedding-search.png")


# ------------------------------------------------------------------ stage 3
def stage3():
    rows = tsv("phase1c_stage3.tsv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, size in zip(axes, ("384", "800")):
        for text, col in (("live", BLUE), ("cached", ACCENT)):
            pts = sorted((int(r["model"].split("_p")[1].split("_")[0]), f(r, "gpu_ms")) for r in rows
                         if r["input"].endswith("x%s" % size) and r["model"].endswith(text))
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=col, marker="o", ms=7, lw=2,
                    markeredgecolor="white", markeredgewidth=2, label="text %s" % text)
            ax.annotate("%.1f" % pts[-1][1], pts[-1], textcoords="offset points",
                        xytext=(6, 6 if text == "live" else -12), fontsize=9.5, color=INK)
        ax.set_xscale("log"); ax.set_xticks([1, 10, 80]); ax.set_xticklabels(["1", "10", "80"])
        ax.set_xlabel("phrases in the prompt")
        ax.set_title("%s %s" % ("crop" if size == "384" else "full frame", size), loc="left", fontsize=11)
        ax.set_ylim(0, None)
        grid(ax, "y")
    axes[0].set_ylabel("GPU ms per request")
    axes[1].legend(frameon=False, loc="lower right")
    fig.suptitle("Grounding DINO-T: ~10 ms a crop, ~30 ms a frame; phrases nearly free up to 10",
                 x=0.01, ha="left", fontsize=12)
    save(fig, "stage3.png")


# ------------------------------------------------------------------ the line (sweeps)
def line_a():
    rows = tsv("phase2_sweepA.tsv")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, k in zip(axes, ("1", "4", "16")):
        for mode, col, lab, ls, lw, ms in (("off", BLUE, "no batcher", "-", 3.2, 9),
                                          ("db0", ACCENT, "dynamic batcher, 0 us (dashed: coincides)", "--", 1.8, 5)):
            pts = sorted((int(r["streams"]), f(r, "fast_ms_p99"), r["overloaded"]) for r in rows
                         if r["k"] == k and r["batching"] == mode)
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=col, marker="o", ms=ms, lw=lw, ls=ls,
                    markeredgecolor="white", markeredgewidth=1.5, label=lab, zorder=3 if mode == "db0" else 2)
            for x, yv, ov in pts:
                if ov == "yes":
                    ax.scatter([x], [yv], s=160, facecolor="none", edgecolor=col, linewidth=1.5, zorder=4)
        ax.axhline(33.3, color=GREY, ls="--", lw=1.2)
        ax.text(1, 33.3, "one frame period (33 ms)", va="bottom", fontsize=9, color=MUTED)
        ax.set_yscale("log"); ax.set_xscale("log", base=2)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: "%g" % v))
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xticks([1, 4, 8, 16]); ax.set_xticklabels(["1", "4", "8", "16"])
        ax.set_xlabel("cameras at 30 fps")
        ax.set_title("K = %s crops per frame" % k, loc="left", fontsize=11)
        grid(ax, "y")
    axes[0].set_ylabel("fast path p99, ms (log)")
    axes[0].legend(frameon=False, loc="upper left", bbox_to_anchor=(0, 0.9), fontsize=9)
    fig.suptitle("Fast-path tail latency as cameras are added (ringed: overloaded, < 95% of frames delivered)",
                 x=0.01, ha="left", fontsize=12)
    save(fig, "line-a.png")


def line_b():
    rows = tsv("phase2_sweepB.tsv")
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.2), sharex=True)
    for col_i, s3 in enumerate(("crop", "frame")):
        for row_i, cams in enumerate(("4", "8")):
            ax = axes[row_i][col_i]
            for prio, col, lab in (("off", BLUE, "priority off"), ("on", ACCENT, "stages 1-2 at PRIORITY_MAX")):
                pts = sorted((float(r["p"]) * 100, f(r, "fast_ms_p99"), f(r, "skip_pct")) for r in rows
                             if r["s3_input"] == s3 and r["streams"] == cams and r["priority"] == prio)
                ax.plot([p[0] for p in pts], [p[1] for p in pts], color=col, marker="o", ms=7 if prio == "off" else 5,
                        lw=2.6 if prio == "off" else 1.6, ls="-" if prio == "off" else "--",
                        markeredgecolor="white", markeredgewidth=1.5, label=lab)
                for x, yv, sk in pts:
                    if sk > 0 and prio == "off":
                        ax.annotate("%.0f%% skipped" % sk, (x, yv), textcoords="offset points", xytext=(4, 6),
                                    fontsize=8.5, color=MUTED)
            ax.set_xscale("symlog", linthresh=1)
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: "%g" % v))
            ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            ax.axhline(33.3, color=GREY, ls="--", lw=1.2)
            ax.text(0.02, 33.3, "one frame period (33 ms)", transform=ax.get_yaxis_transform(), va="bottom",
                    fontsize=8.5, color=MUTED)
            ax.set_xticks([0, 1, 5, 20]); ax.set_xticklabels(["0", "1", "5", "20"])
            ax.set_title("%s cameras, stage 3 on the %s" % (cams, "crop (384)" if s3 == "crop" else "frame (800)"),
                         loc="left", fontsize=11)
            grid(ax, "y")
            if row_i == 1:
                ax.set_xlabel("flag rate p, % of crops sent to stage 3")
            if col_i == 0:
                ax.set_ylabel("fast path p99, ms")
    axes[0][1].legend(frameon=False, loc="upper left", fontsize=9.5)
    fig.suptitle("Does the rare heavy stage slow the fast path? (K = 4; fast-path p99 above 33 ms means the line is falling behind)", x=0.01, ha="left", fontsize=12)
    save(fig, "line-b.png")


def line_c():
    rows = tsv("phase2_sweepC.tsv")
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.2), sharex=True)
    xs = {"0.0": 0, "0.1": 1, "0.2": 2, "0.4": 3}
    for col_i, s3 in enumerate(("crop", "frame")):
        for row_i, cams in enumerate(("4", "8")):
            ax = axes[row_i][col_i]
            for p, col, lab in (("0.05", BLUE, "flag rate 5%"), ("0.2", ACCENT, "flag rate 20%")):
                pts = sorted((xs[r["budget"]], f(r, "fast_ms_p99"), r["overloaded"]) for r in rows
                             if r["s3_input"] == s3 and r["streams"] == cams and r["p"] == p)
                ax.plot([q[0] for q in pts], [q[1] for q in pts], color=col, marker="o", ms=7, lw=2,
                        markeredgecolor="white", markeredgewidth=1.5, label=lab)
                for x, yv, ov in pts:
                    if ov == "yes":
                        ax.scatter([x], [yv], s=170, facecolor="none", edgecolor=col, linewidth=1.5, zorder=4)
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: "%g" % v))
            ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            ax.axhline(33.3, color=GREY, ls="--", lw=1.2)
            ax.text(0.02, 33.3, "one frame period (33 ms)", transform=ax.get_yaxis_transform(), va="bottom",
                    fontsize=8.5, color=MUTED)
            lo, hi = ax.get_ylim()
            ax.set_ylim(min(lo, 20), hi)   # keep the 33 ms line inside every panel
            ax.set_xticks([0, 1, 2, 3]); ax.set_xticklabels(["none", "10%", "20%", "40%"])
            ax.set_title("%s cameras, stage 3 on the %s" % (cams, "crop (384)" if s3 == "crop" else "frame (800)"),
                         loc="left", fontsize=11)
            grid(ax, "y")
            if row_i == 1:
                ax.set_xlabel("stage-3 budget, share of GPU time")
            if col_i == 0:
                ax.set_ylabel("fast path p99, ms")
    axes[0][0].legend(frameon=False, loc="upper right", fontsize=9.5)
    fig.suptitle("A GPU-time budget for stage 3 keeps the line whole, if it fits the headroom (ringed: overloaded)",
                 x=0.01, ha="left", fontsize=12)
    save(fig, "line-c.png")


def line_d():
    rows = tsv("phase2_sweepD.tsv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, k in zip(axes, ("4", "16")):
        for arm, col, lab in (("client", BLUE, "client-driven (2 gRPC calls, post on the client)"),
                              ("bls", ACCENT, "inside Triton (1 request, Python BLS)")):
            pts = sorted((int(r["streams"]), f(r, "fast_ms_p99"), r["overloaded"]) for r in rows
                         if r["k"] == k and r["arm"] == arm)
            ax.plot([q[0] for q in pts], [q[1] for q in pts], color=col, marker="o", ms=7, lw=2,
                    markeredgecolor="white", markeredgewidth=1.5, label=lab)
            for x, yv, ov in pts:
                if ov == "yes":
                    ax.scatter([x], [yv], s=170, facecolor="none", edgecolor=col, linewidth=1.5, zorder=4)
        ax.axhline(33.3, color=GREY, ls="--", lw=1.2)
        ax.text(0.02, 33.3, "one frame period (33 ms)", transform=ax.get_yaxis_transform(), va="bottom",
                fontsize=8.5, color=MUTED)
        ax.set_yscale("log"); ax.set_xscale("log", base=2)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: "%g" % v))
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xticks([1, 4, 8, 16]); ax.set_xticklabels(["1", "4", "8", "16"])
        ax.set_xlabel("cameras at 30 fps")
        ax.set_title("K = %s crops per frame" % k, loc="left", fontsize=11)
        grid(ax, "y")
    axes[0].set_ylabel("fast path p99, ms (log)")
    axes[0].legend(frameon=False, loc="upper left", fontsize=9)
    fig.suptitle("Moving the pipeline into Triton: slower at every load, and it saturates sooner (ringed: overloaded)",
                 x=0.01, ha="left", fontsize=12)
    save(fig, "line-d.png")


if __name__ == "__main__":
    encoders(); big_models(); patchcore_search(); embedding_search(); stage3()
    if os.path.exists(os.path.join(RES, "phase2_sweepA.tsv")):
        line_a()
    if os.path.exists(os.path.join(RES, "phase2_sweepB.tsv")):
        line_b()
    if os.path.exists(os.path.join(RES, "phase2_sweepC.tsv")):
        line_c()
    if os.path.exists(os.path.join(RES, "phase2_sweepD.tsv")):
        line_d()
