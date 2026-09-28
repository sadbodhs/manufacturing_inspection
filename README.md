# Manufacturing inspection: what a label-free visual QA line costs to serve

Measured on one RTX 3090 with TensorRT FP16 and Triton 24.12. Inference only:
no training and no accuracy claims. Models are timed with random weights,
because GPU time depends on a network's structure, not on what it learned.

A companion to [sadbodhs/computer_vision_optimization](https://github.com/sadbodhs/computer_vision_optimization),
which measures the serving layer for a single detector. This repo asks what
changes on an inspection line: several parts per frame, an anomaly model on
every part, and a rare, heavy open-vocabulary model on the flagged ones.

**Status:** in progress. Each phase's predictions are committed before it runs.

| Phase | Question | Status |
|---|---|---|
| 1-pre | Which stage-2 anomaly encoders are fast enough? | done |
| 0 | Do big models gain from batching? | pre-registered |
| 1b | Send the anomaly map back, or reduce it in the graph? | planned |
| 1a | PatchCore: backbone vs memory-bank search | planned |
| 1c | Stage 3 (Grounding DINO) on TensorRT | planned |
| 2 | The whole pipeline under live camera load | planned |

## Layout

- `scripts/trt_bench.py`: the shared harness (export, build once, 3 interleaved timing rounds, medians)
- `scripts/run_phase.sh`: runs a phase in the `triton-server` container under the GPU lock
- `scripts/gpu_lock.sh`: the machine-wide GPU lock, shared with the other studies on this box
- `results/`: one median TSV and one raw TSV per phase

Phase 1-pre (`scripts/inspection_encoders.py`) was first run and published in
the companion repo (commits `4400d76`, `03ed4bf`); its script and results are
copied here unchanged.
