#!/bin/bash
# Manufacturing inspection, Phase 1-pre: stage-2 encoder speed check.
# Holds the GPU lock, runs scripts/inspection_encoders.py inside triton-server,
# and copies the two result files out. Exports and engines stay in the
# container's /tmp - never under triton/models (Triton's model repository).
#
# Usage: scripts/inspection_encoders.sh [model ...]     # default: all six
# Output: results/v3/inspection_encoders.tsv (medians), inspection_encoders_raw.tsv
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
C=triton-server
W=/tmp/insp_enc

"$ROOT/scripts/gpu_lock.sh" acquire inspection "Phase 1-pre encoder speed check"
trap '"$ROOT/scripts/gpu_lock.sh" release inspection' EXIT

docker exec $C rm -rf $W
docker exec $C mkdir -p $W
docker cp "$ROOT/scripts/inspection_encoders.py" $C:$W/
docker exec $C python3 $W/inspection_encoders.py $W $W/raw.tsv $W/summary.tsv "$@"
docker cp $C:$W/raw.tsv "$ROOT/results/v3/inspection_encoders_raw.tsv"
docker cp $C:$W/summary.tsv "$ROOT/results/v3/inspection_encoders.tsv"
docker exec $C rm -rf $W
column -t -s $'\t' "$ROOT/results/v3/inspection_encoders.tsv" | cut -c1-220
