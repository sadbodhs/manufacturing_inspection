#!/bin/bash
# Phase 5: what each embedding-search method costs the GPU (memory, time, utilisation).
# Runs in a throwaway container (mi-search, triton-bench:v3 + FAISS, cuVS, TensorRT
# Python 10.7). Prep builds features, exact references and TensorRT engines once;
# the driver then runs every (method, bank) in a fresh process, 3 shuffled repeats.
#
# Usage: scripts/phase5_run.sh            (prep must have run: scripts/phase5_prep.py)
# Output: results/phase5_footprint_raw.jsonl
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
"$ROOT/scripts/gpu_lock.sh" acquire inspection phase5_footprint
trap '"$ROOT/scripts/gpu_lock.sh" release inspection' EXIT
docker exec mi-search python3 /mi/scripts/phase5_driver.py /mi/results/phase5_footprint_raw.jsonl "$@"
