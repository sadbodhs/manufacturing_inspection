# Map or score — send the anomaly map back, or reduce it in the graph?

A stage-2 model with a dense output (a per-pixel anomaly map, or a reconstructed
image) can hand it back three ways:

- **full**: the full-resolution output in FP32, as the method's reference code returns
  it; the client computes the score
- **FP16 binding**: the same output through an FP16 output binding: half the bytes, no
  change to the graph
- **reduce**: the score and a 1/8-resolution map computed *inside* the engine, so only
  a few floats per crop leave it

This is the companion study's [moving fewer bytes](https://sadbodhs.github.io/computer_vision_optimization/fewer-bytes/)
question, asked of inspection models.

Script: [`phase1b_dense_output.py`](https://github.com/sadbodhs/manufacturing_inspection/blob/main/scripts/phase1b_dense_output.py) ·
data: [`phase1b_dense_output.tsv`](https://github.com/sadbodhs/manufacturing_inspection/blob/main/results/phase1b_dense_output.tsv)

---

## The answer: it barely matters

Total per crop (GPU + transport), median of 3 interleaved repeats:

| | Full FP32 | FP16 binding | Reduce in graph |
|---|---:|---:|---:|
| EfficientAD-S, 256, batch 1 | 1.054 ms | 1.046 ms | 1.044 ms (−1.0%) |
| EfficientAD-S, 256, batch 8 | 0.894 ms | 0.876 ms | 0.879 ms (−1.7%) |
| EfficientAD-S, 512, batch 1 | 4.023 ms | 3.951 ms | 3.939 ms (−2.1%) |
| EfficientAD-S, 512, batch 8 | 3.545 ms | 3.607 ms (**slower**) | 3.493 ms (−1.5%) |
| U-Net reconstruction, 256, batch 1 | 0.732 ms | **0.716 ms (−2.2%)** | 0.741 ms (+1.3%, **slower**) |
| U-Net reconstruction, 256, batch 8 | 0.318 ms | **0.302 ms (−5.0%)** | 0.304 ms (−4.5%) |

**EfficientAD-S** gains at most 2.1%. Its method already emits a one-channel map
(0.26 MB at 256), so there was little to remove, and skipping the full-resolution
upsample saves no measurable compute.

**The reconstruction** is the more instructive case. Reducing in the graph removes
86–97% of its copy-back, as expected. But the error map it has to compute first costs
0.04 ms of GPU time. At batch 1 that outweighs the transport saved, and the reduced
engine is *slower*. At batch 8 it wins 4.5%, but the plain FP16 binding wins 5.0%
without touching the graph.

The FP16 binding has its own surprise: at 512 batch 8 it added 0.65 ms of GPU time to
EfficientAD-S and made it 1.7% slower overall. That engine was not profiled, so the
cause is not established here. The lesson stands without it: an output-format change
is not free by construction; measure it.

**Where the lever does pay** is an output that is large *features*, not a map.
PatchCore's backbone emits 6–25 MB per crop, and there reducing inside the engine is
worth 22–35% of the whole cost: see [PatchCore's search](patchcore-search.md).

## What was predicted

| | Prediction | Measured | |
|---|---|---|---|
| P1 | For EfficientAD-S, reducing saves ≤ 3% at 256 b1 and ≤ 5% anywhere | −1.0 to −2.1% | **held** |
| P2 | The FP16 binding halves D2H; for EfficientAD-S ≤ 0.01 ms per crop | D2H −35 to −49%; but +0.65 ms GPU at 512 b8 | **mostly held** |
| P3 | For reconstruction, reducing removes ≥ 90% of D2H, ~4% (b1) and ~10% (b8) per crop | removes 86% / 97%; +1.3% (b1) and −4.5% (b8) | **failed** on the payoff |
