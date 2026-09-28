# Manufacturing inspection on one GPU

*What a label-free visual inspection line costs to serve, measured on an RTX 3090
with TensorRT and Triton.*

A companion to [the serving study](https://sadbodhs.github.io/computer_vision_optimization/),
which measures how to serve **one detector** on camera frames. An inspection line is a
different shape of problem. Every part is photographed. Almost every part is good. The
rare bad one has to be caught, **and nobody has labelled what "bad" looks like**,
because the defect that matters next month may not exist yet.

That changes what the GPU is asked to do. There are several parts per frame, not one;
a second model runs on every one of them; and a third, much heavier model runs rarely
and unpredictably. This site measures what that costs, stage by stage and then as a
whole line under live camera load.

![The three-stage inspection pipeline: a line camera feeds yolov8s, which crops each part; an anomaly model trained on good parts scores every crop; a fraction p of flagged crops goes asynchronously to an open-vocabulary model that names or outlines the defect. Tags show which page measures each part.](img/pipeline.svg)

**Inference only.** No training, no fine-tuning, no claims about detecting defects.
GPU time depends on a network's structure, not on what it learned, so models are timed
with random weights, and [that premise is tested](big-models.md#weights-do-not-matter-the-export-does)
rather than assumed. Anything that *would* depend on accuracy, such as how often a part
is flagged, is a setting that is varied instead.

---

## Why three stages, and why no defect labels

The obvious design is a detector trained on defect classes. It fails on both
properties that define inspection. Defects are **rare**, often under 1% of parts, so
each class has few examples and a detector learns mostly "good". And defects are
**open-ended**: a new failure mode appears, and a supervised model is out of date the
day it does. So the pipeline needs **no defect labels at all**:

| Stage | Job | Runs on | Trained on | Why this shape |
|---|---|---|---|---|
| 1. Locate | find and crop each part | every frame | parts, which are in every image | no class imbalance: every image has parts |
| 2. Judge | score how unlike a good part each crop is | every crop | **good parts only** | the one thing a line has in abundance |
| 3. Explain | name or outline what was flagged | a fraction *p* of flagged crops | nothing (open vocabulary) | a new defect type is a new text prompt, not a new model |

The serving consequences are the subject of this site. Stage 2 runs *K* times per
frame, so it is a batch by construction. Stage 3 is rare, heavy and bursty, and must
not stall stages 1–2, which keep the line moving.

## Choosing the models

The rule was **few models, each answering a different serving question**.

**Stage 1: yolov8s, already measured.** 0.99 ms of engine time at 640 and 1.25 ms
inside a full pipeline, in [the serving study](https://sadbodhs.github.io/computer_vision_optimization/results/).
A locator does not need high resolution, since parts are large in the frame; on a
fixtured line it can be a fixed crop and cost nothing. **Resolution matters for
stage 2**, which is looking for scratches, so that is where input size is varied.

**Stage 2: one encoder per family of anomaly method.**

| Family | How it judges a part | Chosen |
|---|---|---|
| Memory bank | stores good-part patch features; a patch far from every stored one is anomalous | **PatchCore** on WideResNet-50-2, the standard baseline |
| Student–teacher | a student imitates a teacher on good parts; where it fails, the part is unusual | **EfficientAD-S**, designed for millisecond latency |
| Foundation features | the memory-bank idea on a general-purpose ViT's features | **AnomalyDINO** on DINOv2 ViT-S/B |
| Light CNN features | Gaussian or flow models over a small backbone (PaDiM, STFPM, FastFlow) | the **cheap floor**: ResNet-18 |
| Reconstruction | rebuild the part; anomalies reconstruct badly (DRAEM) | **reference only**: U-Net R34 |

Deliberately **not** here: normalizing flows (they run on the same backbones, so their
extra cost is a small head), diffusion detectors (many denoising steps per image, a
different budget from a per-part check), and vision-language models as the judge (they
belong in stage 3 if anywhere: a line cannot afford one on every part).

**Stage 3: an open-vocabulary detector, only on flagged crops.** Grounding DINO (tiny)
names defects from text prompts. SAM outlines a region once it has a box. YOLO-World
was the fallback in case Grounding DINO would not convert to TensorRT; it was not
needed.

## What was measured, and what it found

Every experiment was pre-registered: its predictions were committed before it ran,
and each page scores them, including the ones that failed.

| Page | Question | Finding |
|---|---|---|
| [Which encoders](encoders.md) | Which stage-2 encoders are fast enough? | At a 256 crop all cost 0.29–1.35 ms, about the cost of finding the part. EfficientAD-S has the most expensive network of the CNNs; its speed is its method. |
| [Map or score](dense-output.md) | Send the anomaly map back, or reduce it in the graph? | Barely matters: ≤ 2% for EfficientAD-S, and for reconstruction a plain FP16 output does better than reducing in the graph. |
| [PatchCore's search](patchcore-search.md) | Backbone or memory-bank search? | Below ~8k patches, searching inside the engine is *cheaper* than not searching, because it avoids moving 6–25 MB of features. Past ~20k the bank sets the cost. |
| [Faster embedding search](embedding-search.md) | Do FAISS or cuVS search the bank faster? | No. At PatchCore's query shape, TensorRT's brute force beats every index with recall ≥ 0.9 by 4× or more. |
| [Do big models batch](big-models.md) | Does batching flagged crops help stage 3? | It depends on the architecture, not the size: RT-DETR saves 38%, SAM-B gets 14% *slower*. Random weights time the same as pretrained; the export path moves SAM by 9%. |
| [Grounding DINO on TensorRT](stage3.md) | Does stage 3 convert, and what does it cost? | Converts with one workaround. ~10 ms per crop, ~30 ms per frame; phrases are nearly free up to 10; caching the text saves 0.5 ms. |
| [Under live load](the-line.md) | The whole line with virtual 30 fps cameras | One 3090 inspects ~10 cameras at 4 parts per frame, within 10% of the engine arithmetic. Triton's batcher adds nothing; a frame's crops are already the batch. Stage 3 hurts twice: a 30 ms execution sets the fast path's tail, and past the GPU budget the line collapses. Stream priority trims the tail by a quarter to a third and cannot prevent the collapse. |

How it was measured, and how to reproduce it: [method](method.md).
