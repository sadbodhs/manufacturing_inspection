#!/bin/bash
# Phase 1a-2: embedding search for PatchCore's memory bank.
# Runs in a throwaway container (mi-search) from the same triton-bench:v3 image,
# so FAISS / cuVS never touch the Triton containers. FAISS and cuVS ship CUDA 12
# wheels; the image's torch is CUDA 13, and the two runtimes coexist as separate
# libraries. cuVS is best-effort: if it does not install, the phase reports that.
#
# Usage: scripts/phase1a2_search.sh
# Output: results/phase1a2_search.tsv, results/phase1a2_env.json, results/phase1a2_install.log
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
C=mi-search

docker rm -f $C >/dev/null 2>&1 || true
docker run -d --name $C --gpus all --network host \
  -v /home/suchi/coco:/coco:ro -v "$ROOT:/mi" triton-bench:v3 sleep infinity >/dev/null
trap 'docker rm -f $C >/dev/null 2>&1 || true' EXIT

{
  docker exec $C pip install -q faiss-gpu-cu12 2>&1 | tail -2
  docker exec $C pip install -q --extra-index-url=https://pypi.nvidia.com cuvs-cu12 cupy-cuda12x 2>&1 | tail -2 \
    || echo "cuVS install failed"
  docker exec $C python3 -c "import faiss; print('faiss', faiss.__version__, 'gpus', faiss.get_num_gpus())"
  docker exec $C python3 -c "from cuvs.neighbors import cagra; print('cuvs ok')" || echo "cuVS import failed"
} > "$ROOT/results/phase1a2_install.log" 2>&1
cat "$ROOT/results/phase1a2_install.log"

"$ROOT/scripts/gpu_lock.sh" acquire inspection phase1a2_search
trap '"$ROOT/scripts/gpu_lock.sh" release inspection; docker rm -f $C >/dev/null 2>&1 || true' EXIT
docker exec $C python3 /mi/scripts/phase1a2_search.py /tmp/out
docker cp $C:/tmp/out/phase1a2_search.tsv "$ROOT/results/"
docker cp $C:/tmp/out/phase1a2_env.json "$ROOT/results/"
column -t -s $'\t' "$ROOT/results/phase1a2_search.tsv"
