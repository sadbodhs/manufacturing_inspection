# Manufacturing inspection: what a label-free visual QA line costs to serve

**📖 [Read it as a site](https://sadbodhs.github.io/manufacturing_inspection/)**

Measured on one RTX 3090 with TensorRT 10.7 and Triton 24.12. Inference only: no
training and no claims about finding defects. Models are timed with random weights,
because GPU time depends on a network's structure, not on what it learned, and that
premise was tested rather than assumed.

A companion to [sadbodhs/computer_vision_optimization](https://github.com/sadbodhs/computer_vision_optimization),
which measures the serving layer for a single detector. This repo asks what changes
on an inspection line: several parts per frame, an anomaly model trained on good
parts only on every one of them, and a rare, heavy open-vocabulary model on the
flagged ones.

## What it found

| Question | Finding |
|---|---|
| Which stage-2 encoders are fast enough? | At a 256 crop all cost 0.29–1.35 ms. EfficientAD-S runs the most expensive CNN; its speed is its method. |
| Send the anomaly map back, or reduce it in the graph? | Barely matters (≤ 2%); a plain FP16 output beats reducing in the graph for reconstruction. |
| PatchCore: backbone or memory-bank search? | Below ~8k patches, searching inside the engine is cheaper than not searching; past ~20k the bank sets the cost. |
| Do FAISS or cuVS search the bank faster? | No: TensorRT's brute force beats every index with recall ≥ 0.9 by 4× or more. |
| What does each search method cost the GPU? | TensorRT brute force is lightest up to ~100k patches; FAISS adds 1.55 GB of scratch; at 1M IVF-PQ holds 2.0 GB and CAGRA stays at 16 ms; nvidia-smi utilization cannot tell them apart. |
| Do big models gain from batching? | By architecture, not size: RT-DETR −38%, SAM-B +14%. Weights do not change timing; the export path moved SAM by 9%. |
| Does Grounding DINO convert, and what does it cost? | One workaround; ~10 ms a crop, ~30 ms a frame. |
| The whole line, live | ~10 cameras per 3090 at 4 parts per frame, within 10% of the arithmetic. Stage 3 sets the tail and, past the GPU budget, collapses the line; priority does not prevent it. |
| Can stage 3 be kept from collapsing the line? | A GPU-time budget within the headroom (10% at 8 cameras) keeps every frame flowing; it cannot fix the tail. |
| Is the pipeline faster inside Triton (Python BLS)? | No: 1.2 ms slower per frame and 22% less capacity than driving it from the client. |

Every phase's predictions were written before it ran and committed before its
measurement (one was committed after a smoke run, and says so); the pages score them.

## Layout

- `scripts/trt_bench.py`: the shared engine harness (export, build once, 3 interleaved timing rounds, medians)
- `scripts/run_phase.sh`: runs an engine phase in the `triton-server` container under the GPU lock
- `scripts/phase*.py|sh`: one script per phase, predictions in its header
- `cpp/inspect_client.cu`: the live-line client (virtual 30 fps cameras, all three stages)
- `scripts/phase2_setup.sh`, `phase2_run.sh`: the second Triton server and the sweeps
- `scripts/plot_all.py`: every figure on the site, from `results/`
- `results/`: one median TSV and one raw file per phase
- `docs/`: the site ([method and reproduction](https://sadbodhs.github.io/manufacturing_inspection/method/))

The encoder check (`scripts/inspection_encoders.py`) was first run in the companion
repo (commits `4400d76`, `03ed4bf`); its script and results are copied here unchanged.
