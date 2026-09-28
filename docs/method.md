# Method — how these numbers were produced

The same bar as [the serving study](https://sadbodhs.github.io/computer_vision_optimization/methodology/):
predictions committed before each run, repeated and interleaved measurements, an
exclusive GPU, and every result in the repository with the script that made it.

## Hardware and software

One NVIDIA RTX 3090 (24 GB, compute capability 8.6), TensorRT 10.7 and Triton 24.12,
everything in Docker (the `triton-bench:v3` image of the companion repo). FP16
engines throughout.

## Engine timings (every page except embedding search and the live line)

[Faster embedding search](embedding-search.md) times library calls instead (FAISS,
cuVS, PyTorch): median of 50 synchronised calls after 10 warm-ups, and the whole phase
was run twice, with rows agreeing within 7% between the runs.

A shared harness, [`trt_bench.py`](https://github.com/sadbodhs/manufacturing_inspection/blob/main/scripts/trt_bench.py),
treats every phase the same way:

1. **Export** each model to ONNX, in its own directory.
2. **Build** each engine once with `trtexec --fp16`.
3. **Time** in 3 interleaved rounds: every engine once per round, in a rotated order,
   with `trtexec`'s default timing window. Interleaving means a slow minute on the
   machine cannot land on one configuration only.
4. **Report** the median of the 3 rounds, and the spread. Repeats agreed within
   1.5% on every phase, and usually within 0.5%.

`trtexec` reports three components per inference: **H2D** (input copy to the GPU),
**GPU compute** (the engine) and **D2H** (output copy back). "Total" on these pages
is their sum; "transport" is H2D + D2H. These are engine-level numbers: the cost of
the model, not of a serving stack around it. The [live line](the-line.md) measures the
stack.

## Random weights

GPU time depends on a network's structure, not on what it learned, so models are
built from their architecture with random weights (no downloads, no training). This
was **tested, not assumed**: the model zoo's own SAM-B graph timed with its pretrained
weights and with every weight randomized differ by 0.2%
([details](big-models.md#weights-do-not-matter-the-export-does)). The same test found
that the *export path* can move an engine by 9%, which is why every phase uses one
shared export harness.

The one exception is [faster embedding search](embedding-search.md), which needs real
features to measure recall, and uses pretrained WideResNet-50-2 features of COCO
images for that reason.

## The live line

The [whole-line sweeps](the-line.md) run a C++ client,
[`inspect_client.cu`](https://github.com/sadbodhs/manufacturing_inspection/blob/main/cpp/inspect_client.cu),
built on the companion repo's paced client. Each thread is one **virtual 30 fps
camera**: frame *k* is due at a fixed time whether or not the server has kept up, and
every latency is measured **from that due time**. A slow server cannot hide its delay
by making the client send later (coordinated omission). Frames are the companion
repo's preprocessed replay frames, uploaded into CUDA shared memory as a camera's frame
would arrive. Each configuration runs 15 s after a 1 s warm-up, 3 times, in a seeded
interleaved order, with a pause between runs so no run inherits another's queue.

It is served by a second Triton server with its own model repository, so nothing is
added to the companion repo's server; both stayed loaded on the GPU throughout, and
the companion's was idle.

## Exclusive GPU

The machine also runs other studies. Two workloads on one GPU corrupt each other's
numbers: the companion study measured sharing at about 24% error, bigger than most
effects reported here. Every GPU run holds a machine-wide lock,
[`gpu_lock.sh`](https://github.com/sadbodhs/manufacturing_inspection/blob/main/scripts/gpu_lock.sh),
shared with the other studies.

## Pre-registration

Each phase's predictions were written into its script and committed before the phase
ran; the commit history shows the order. Every page scores its predictions, and most
of them failed in some way. One pre-registration was committed late: [Inside Triton](bls.md)'s predictions were
written before its model ran, but committed after one smoke run that already
contradicted one of them; they were committed unchanged, and the commit says so.
Four corrections were made during the work and are recorded where they happened: the recall reference on
[faster embedding search](embedding-search.md#a-correction-made-during-the-run), the
mask workaround bug on [Grounding DINO](stage3.md#it-converts-with-one-workaround),
the export-versus-weights confound on [big models](big-models.md#weights-do-not-matter-the-export-does),
and an explanation for the BLS slowdown that a follow-up sweep refuted
([inside Triton](bls.md#correction-it-is-not-the-cuda-contexts)).

## Reproduce

```bash
scripts/run_phase.sh phase0_big_models.py phase0_batching      # any engine phase
scripts/phase0b_sam_check.sh                                    # needs the zoo's SAM-B ONNX
scripts/phase1a2_search.sh                                      # FAISS / cuVS container
scripts/phase2_setup.sh && scripts/phase2_run.sh A B            # the live line
python3 scripts/phase2_summarize.py
```

Engines, ONNX files and replay frames are not committed: they are GPU-specific or
large, and every script above regenerates what it needs.

## What this does not cover

- **Accuracy.** Nothing here says how well any method finds defects.
- **Other GPUs.** Absolute numbers are for one RTX 3090; shapes (what scales, what
  batches, where the crossover sits) are more likely to carry over than positions.
- **Training.** Memory banks are random stand-ins of realistic size; building a real
  coreset is a training-time cost not measured here.
