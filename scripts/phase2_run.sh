#!/bin/bash
# Phase 2: run sweeps against the mi-triton server under the GPU lock.
#   scripts/phase2_run.sh A B        # both sweeps, in order
# Needs scripts/phase2_setup.sh to have run (server up, client built).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
curl -sf localhost:8100/v2/health/ready >/dev/null || { echo "mi-triton is not ready; run scripts/phase2_setup.sh"; exit 1; }
"$ROOT/scripts/gpu_lock.sh" acquire inspection "phase2 sweeps: $*"
trap '"$ROOT/scripts/gpu_lock.sh" release inspection' EXIT
for s in "$@"; do
  python3 "$ROOT/scripts/phase2_sweep.py" "$s"
done
