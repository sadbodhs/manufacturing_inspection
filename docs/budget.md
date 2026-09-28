# Budgeting stage 3 — admission control by GPU time

[Under live load](the-line.md) found that the rare heavy stage can collapse the
whole line, and that neither of the obvious guards stops it. A pool of stage-3
slots never filled at 8 cameras, because the fast path slowed first. Stream priority
cannot interrupt a running kernel. This page tests the guard that follows from that
diagnosis: **give stage 3 a fixed share of GPU time**, and shed flags beyond it.

The mechanism is a token bucket shared by every camera, filled with milliseconds of
GPU time at the budget rate (a 10% budget refills 100 ms of GPU time per second).
Each flagged crop must pay its measured stage-3 cost before it is sent: 9.43 ms for a
384 crop, 29.60 ms for an 800 frame, the medians from
[Grounding DINO on TensorRT](stage3.md). If it cannot pay, it is **shed and counted**.
Unlike the slot pool, this bounds stage 3's share of the GPU whatever the fast path is
doing.

*K* = 4 parts per frame, priority off, 4 or 8 cameras, flag rate 5% or 20%, stage 3
on the crop or the frame, budget none / 10% / 20% / 40%; 96 runs, 3 repeats each.

Client: [`inspect_client.cu`](https://github.com/sadbodhs/manufacturing_inspection/blob/main/cpp/inspect_client.cu)
(`--s3-budget`) · data: [`phase2_sweepC.tsv`](https://github.com/sadbodhs/manufacturing_inspection/blob/main/results/phase2_sweepC.tsv)

---

## A budget inside the headroom keeps the line whole

From [under live load](the-line.md), the fast path alone keeps the GPU about 41% busy
at 4 cameras and about 82% at 8. So at 8 cameras there is roughly 15% of GPU time to
spare, and the budget has to fit inside it.

![Fast-path p99 against the stage-3 budget for 4 and 8 cameras, crop and frame input, at flag rates of 5% and 20%. Without a budget, every panel except 4 cameras with crops is overloaded. A 10% budget brings every panel back near or under one frame period. At 8 cameras, 20% is at the edge and 40% collapses again.](img/line-c.png)

8 cameras, where the unbudgeted line collapsed (frames delivered, fast-path p99):

| | no budget | 10% | 20% | 40% |
|---|---|---|---|---|
| crop, *p* = 5% | 80%, 3.7 s | **100%, 20.5 ms** | 99.3%, 207 ms | 81%, 3.4 s |
| crop, *p* = 20% | 67%, 6.3 s | **99.9%, 23.4 ms** | 99.2%, 190 ms | 73%, 5.4 s |
| frame, *p* = 5% | 46%, 10.9 s | **100%, 44.1 ms** | 99.7%, 57.5 ms | 78%, 4.4 s |
| frame, *p* = 20% | 35%, 12.7 s | **100%, 44.8 ms** | 98.6%, 269 ms | 74%, 5.2 s |

**A 10% budget delivers every frame at every flag rate**, where no budget delivered
35–80%. The operator's wait for an explanation drops with it: 32–56 ms at the median
instead of 2–8 seconds, because an admitted request no longer queues behind a
backlog. **20% is at the edge**: frames still arrive, but the tail swells to
58–269 ms. **40% collapses again**: the budget exceeds the headroom, so it no longer
protects anything. At 4 cameras, with ~59% headroom, every budget holds.

The rule is short: **stage 3's budget must be at most one minus the fast path's share
of the GPU**, and the fast path's share is exactly what the
[capacity arithmetic](the-line.md#how-many-cameras-one-3090-inspects) predicts.

## What it costs, and what it cannot fix

**The cost is explanations.** At 8 cameras with a 10% budget, stage 3 explains about
10 crops a second, or about 3 whole frames. At a 20% flag rate that sheds up to 98% of
flags. Shedding is the honest outcome: the GPU cannot do more, and the alternative was
a line that stopped. What happens to shed flags is a product decision this page does
not make: queue them for later, explain them on another GPU, or log them unexplained.

**It cannot fix the tail.** With whole frames admitted, the fast path's p99 sits at
26–29 ms at 4 cameras whatever the budget, against 7.7 ms with no stage 3. A budget
limits how *often* stage 3 runs, not how *long* each run holds the GPU, and a frame
that lands behind one 30 ms Grounding DINO execution waits for it. The fix for the
tail is a shorter execution: send the crop, not the frame
([under live load](the-line.md#does-the-rare-heavy-stage-slow-the-fast-path)).

## Size the burst to one frame's crops

With the bucket's cap set to 100 ms of wall time, a 10% budget holds about one crop's
worth of tokens. Flags arrive in bursts (several crops of one frame at once), so at a
low flag rate tokens that overflow the cap are wasted: crops were served 27% below
budget at 4 cameras and 14% below at 8. At *p* = 20%, with a steady supply of flags,
service was within 2–11% of budget ÷ cost.

A follow-up sweep tested the obvious fix, a burst of one frame's crops (*K* × cost),
at 8 cameras and a 10% budget (3 repeats, predictions committed first):

| 8 cameras, 10% budget | default burst | burst = one frame's crops |
|---|---:|---:|
| crop, *p* = 5%: served vs budget ÷ cost | 9.07/s (−14%) | **10.64/s (±0%)** |
| crop, *p* = 20% | −2% | ±0% |
| frame, *p* = 5% / 20% | −5% / −3% | −1% / −1% |
| frames delivered | 99.8–100% | 99.9–100% |
| fast-path p99, crop, *p* = 5% | 20.6 ms | 22.8 ms (+11%) |

**It works**: stage 3 is served exactly on budget, every frame still gets through, and
the fast path's tail grows 11%. A production bucket should size its burst to at least
one frame's crops.

## What was predicted

Written after [under live load](the-line.md), before this sweep ran:

| | Prediction | Measured | |
|---|---|---|---|
| C1 | At 8 cameras a 10% budget prevents the collapse at every *p* and input | ≥ 99.9% of frames delivered in all four | **held** |
| C2 | Above the ~15% headroom the collapse returns: 20% at the edge, 40% collapses; at 4 cameras every budget holds | 20%: 98.6–99.7% delivered, p99 58–269 ms; 40%: 73–81%; 4 cameras all hold | **held** |
| C3 | A budget fixes the collapse, not the tail: with frames admitted, p99 ≥ ~30 ms at 4 cameras | 26–29 ms at every budget | **held in substance** |
| C4 | Stage 3 served at ~budget ÷ cost (within 15%); explain latency in tens of ms | within 2–11% at *p* = 20%, 27% under at *p* = 5%; 17–56 ms | **mostly held** |
| E1 | A burst of one frame's crops brings service at *p* = 5% within 10% of budget ÷ cost, frames ≥ 99.9% delivered, p99 up ≤ 20% | ±0%; 99.9%; +11% | **held** |
| E2 | At *p* = 20% the burst size changes service by < 5% | ≤ 2% | **held** |
