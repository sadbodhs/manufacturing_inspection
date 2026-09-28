#!/bin/bash
# Run one phase's timing script inside the triton-server container, holding
# the GPU lock, and copy its two result files into results/.
#
#   scripts/run_phase.sh <phase_script.py> <result_name> [extra args for the script]
#   e.g. scripts/run_phase.sh phase0_big_models.py phase0_batching
#   -> results/phase0_batching.tsv (medians) + results/phase0_batching_raw.tsv
#
# Work happens in the container's /tmp and is deleted afterwards. Nothing is
# written under the benchmark repo's triton/models (Triton's model repository,
# mounted at /models): a stray directory there takes the server down on its
# next restart.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPT="${1:?phase script}"; NAME="${2:?result name}"
C=triton-server
W=/tmp/mi_$NAME

"$ROOT/scripts/gpu_lock.sh" acquire inspection "$NAME"
trap '"$ROOT/scripts/gpu_lock.sh" release inspection; docker exec $C rm -rf $W || true' EXIT

docker exec $C rm -rf $W
docker exec $C mkdir -p $W
for f in "$ROOT"/scripts/*.py; do docker cp "$f" $C:$W/; done   # phases share model builders
docker exec $C python3 $W/$SCRIPT $W/work $W/raw.tsv $W/summary.tsv "${@:3}"
mkdir -p "$ROOT/results"
docker cp $C:$W/raw.tsv "$ROOT/results/${NAME}_raw.tsv"
docker cp $C:$W/summary.tsv "$ROOT/results/${NAME}.tsv"
# any further TSVs a phase writes beside its summary keep their own names
for f in $(docker exec $C sh -c "ls $W/*.tsv"); do
  case "$(basename "$f")" in raw.tsv|summary.tsv) ;; *) docker cp "$C:$f" "$ROOT/results/";; esac
done
column -t -s $'\t' "$ROOT/results/${NAME}.tsv" | cut -c1-230
