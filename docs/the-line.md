# Under live load — the whole line with virtual 30 fps cameras

Every other page times one model in isolation. This one runs the whole line against
a Triton server, the way a deployment would, with virtual cameras that do not wait
for the answer:

- **Stage 1:** yolov8s at 640, exactly the engine the companion study measured, with
  its GPU post-processing.
- **Crops:** *K* seeded boxes per frame, cut from the frame and resized to 256 on the
  GPU, the same boxes for every configuration.
- **Stage 2:** WideResNet-50-2 PatchCore with the nearest-neighbour search inside the
  engine and a 10k-patch bank ([PatchCore's search](patchcore-search.md)). The *K* crops
  go as **one request of batch K**.
- **Stage 3:** Grounding DINO-T, 10 phrases, text cached ([Grounding DINO](stage3.md)).
  Each crop is flagged with seeded probability *p*; a flagged crop, or the whole frame,
  goes to stage 3 **asynchronously**, so the line never waits for it. If every stage-3
  slot is busy the crop is **skipped and counted**, never silently dropped.

Every latency is measured from the frame's **due time**: a camera exposes frame *k* at
its scheduled moment whether or not the server kept up, so a slow server cannot hide
its delay. "Fast path" is due time to stage-2 result, what keeps the line moving;
"explain" is due time to stage-3 result, what an operator waits for. Each configuration
ran 15 s, 3 times, interleaved. How it was built: [method](method.md#the-live-line).

Client: [`inspect_client.cu`](https://github.com/sadbodhs/manufacturing_inspection/blob/main/cpp/inspect_client.cu) ·
sweeps: [`phase2_sweep.py`](https://github.com/sadbodhs/manufacturing_inspection/blob/main/scripts/phase2_sweep.py) ·
data: [`phase2_sweepA.tsv`](https://github.com/sadbodhs/manufacturing_inspection/blob/main/results/phase2_sweepA.tsv),
[`phase2_sweepB.tsv`](https://github.com/sadbodhs/manufacturing_inspection/blob/main/results/phase2_sweepB.tsv)
(medians) and the `.jsonl` files beside them (every run)

---

## How many cameras one 3090 inspects

The prediction came straight from the engine pages: yolov8s costs 1.0 ms a frame, and
stage 2 at batch *K* costs about 0.9 / 2.7 / 9.6 ms for *K* = 1 / 4 / 16. A camera at
30 fps offers 33.3 ms per frame, so one GPU should hold roughly 17 / 9 / 3 cameras.

![Fast-path p99 against the number of cameras for K = 1, 4 and 16 crops per frame, with and without Triton's dynamic batcher. The two lines coincide everywhere. K = 1 stays under 13 ms at 16 cameras; K = 4 is fine at 8 cameras and overloaded at 16; K = 16 is overloaded from 4 cameras.](img/line-a.png)

| Parts per frame | Predicted | Measured |
|---|---:|---|
| *K* = 1 | ~17 cameras | holds **16** with p99 12 ms (the ceiling was not reached) |
| *K* = 4 | ~9 cameras | **9.8 cameras**: 8 hold with p99 16 ms; 16 deliver 295 of 480 frames/s |
| *K* = 16 | ~3 cameras | **3.1 cameras**: from 4 cameras it delivers ~92 frames/s |

**The engine-level arithmetic predicts the live line within about 10%.** Capacity here
is the frame rate an overloaded configuration still delivers: at *K* = 4 with 16
cameras the GPU completes 295 frames a second, 9.8 cameras' worth, and the rest queue
until latency is measured in seconds. That is the practical use of the engine pages:
add up the per-frame GPU time of each stage, divide it into the frame period, and the
answer is close.

Latency stays low right up to the edge. At *K* = 4, eight cameras (about 80% of the
measured capacity) still get every frame back within 16 ms at p99, half a frame period.

**The dynamic batcher changes nothing.** With and without Triton's dynamic batcher
(0 µs window), every configuration lands within about 3%, including *K* = 1 with 16
cameras, where it had the most to merge. Each camera has one request in flight, and
cameras fire at staggered moments, so stage 2 rarely finds two requests waiting at
once. A frame's *K* crops are already the batch, and that is where the batching
saving from the [encoder page](encoders.md#batching-crops-pays-off-far-more-than-batching-frames)
comes from.

One curiosity: at *K* = 1, four cameras get a lower median than one (3.8 against
4.4 ms). The likely reason is the GPU lowering its clocks between one camera's frames,
33 ms apart; it was not verified here.

## Does the rare heavy stage slow the fast path?

Stage 3 is rare but heavy: one flagged crop costs ~10 ms of GPU (~30 ms on a whole
frame), about three times a frame's entire fast path. Sweep B fixes *K* = 4 and varies
the flag rate *p*, 4 or 8 cameras, whether stage 3 sees the crop (384) or the frame
(800), and whether stages 1–2 run at Triton's `PRIORITY_MAX`. Triton 24.12 does turn
that setting into CUDA stream priority: its log shows the priority instances created
at stream priority −5, the others at 0.

![Fast-path p99 against the flag rate, for 4 and 8 cameras and for stage 3 on the crop or the frame, with and without priority on stages 1–2. At 4 cameras with crops the line holds to 20% with rising tail latency; every other panel crosses the 33 ms frame period and collapses to seconds; priority lowers the curve slightly only where the line is not overloaded.](img/line-b.png)

Fast-path p99 (median of 3), priority off:

| | *p* = 0 | 1% | 5% | 20% |
|---|---:|---:|---:|---:|
| 4 cameras, stage 3 on the crop | 7.8 ms | 9.6 ms | 15.6 ms | 26.3 ms, 34% of flags skipped |
| 4 cameras, stage 3 on the frame | 7.7 ms | **33.1 ms** | overloaded (674 ms p50) | overloaded |
| 8 cameras, stage 3 on the crop | 16.1 ms | 34.2 ms | overloaded (2.1 s p50) | overloaded |
| 8 cameras, stage 3 on the frame | 16.0 ms | overloaded | overloaded | overloaded |

Stage 3 hurts the line in **two different ways**, and they need different fixes.

**1. The tail: one long execution delays the frames behind it.** At 4 cameras with
stage 3 on whole frames, a 1% flag rate is about 4 stage-3 requests a second (57 in
15 s), and the GPU is only about half busy, yet the fast path's p99 goes from 7.7 to 33.1 ms. That is one
frame's fast path plus one ~30 ms Grounding DINO execution it had to wait behind. A
rare heavy request sets the tail of every frame that lands behind it, whatever the
average load. Sending stage 3 the crop instead of the frame cuts that to 9.6 ms, because
the execution in the way is a third as long.

**2. The collapse: total demand exceeds the GPU.** At 8 cameras the fast path alone
keeps the GPU over 80% busy; stage 3 on crops at *p* = 5% adds about 40% more, and the
line falls over: 192 of 240 frames a second, 2 s median delay. The arithmetic is the
same as for capacity: add the fast path's GPU time per second (cameras × 30 × per-frame
time) and stage 3's (cameras × 30 × *K* × *p* × its time), and keep the sum well under
100%.

Two things do **not** protect the line:

- **The stage-3 slot pool.** Skipping flagged crops when every slot is busy did bound
  stage 3 at 4 cameras, *p* = 20% (34% skipped, the line held). But at 8 cameras and
  *p* = 5% no slot was ever full, nothing was skipped, and the line collapsed anyway:
  the fast path slowed first. A slot count is not a GPU budget. Stage 3 needs
  admission control tied to measured GPU time, or its own GPU.
- **Stream priority.** Running stages 1–2 at `PRIORITY_MAX` lowered the fast path's
  p99 where the line was coping: 15.6 → 13.1 ms at 4 cameras, crops, *p* = 5%, and
  33.1 → 26.6 ms with frames at 1%. That recovers a quarter to a third of the rise. Under
  overload it changes almost nothing. Priority decides which kernel starts next; it
  cannot interrupt one that is already running, and Grounding DINO's executions are
  long.

**For a deployment:** send stage 3 the crop, not the frame; cap its share of the GPU by
budget rather than by queue length ([measured on the next page](budget.md): a budget that
fits the headroom keeps the line whole); and treat priority as a tail trim, not a
safety net. If stage 3 must see whole frames, or the flag rate cannot be bounded, give it its
own GPU.

## What was predicted

| | Prediction | Measured | |
|---|---|---|---|
| A1 | Capacity follows the arithmetic within ~20%: K=4 holds 8 and is overloaded at 16; K=16 overloaded from 4; K=1 holds 16 | 9.8 and 3.1 cameras; K=1 holds 16 | **held** |
| A2 | Fast path at 1 camera, p50: ~3 / ~5 / ~11 ms for K = 1 / 4 / 16 | 4.4 / 5.6 / 12.2–12.4 ms | **partly held** (K=1 47% over) |
| A3 | The batcher matters only at K=1 with many cameras | no measurable effect anywhere | **failed** |
| B1 | Stage 3 is the load: at *p* = 20% with 4 cameras the GPU is over-committed and stage 3 skips | 34% skipped, fast path still delivered | **held** |
| B2 | Without priority, fast-path p99 at *p* = 5% (crop, 4 cameras) ≥ 50% above *p* = 0 | +101% (7.8 → 15.6 ms) | **held** |
| B3 | Priority recovers less than half of that rise | 32% at 5%, 12% at 20% | **held** |
| B4 | Frame input costs ~3× a crop, so it skips ~3× as often and hurts the fast path more | hurts far more (collapses where crops hold); skips 52% vs 34%, not 3× | **partly held** |
