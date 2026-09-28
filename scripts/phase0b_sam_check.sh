#!/bin/bash
# Phase 0b: stage the model zoo's SAM-B ONNX into the container, run the check
# under the GPU lock, and collect its four result files.
# The zoo ONNX lives outside both repos (it is not mounted into triton-server).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ZOO_ONNX=/home/suchi/sadbodh/model_exports/_exp/sam_vit_b_enc.onnx
C=triton-server

docker exec $C rm -rf /tmp/mi_inputs
docker exec $C mkdir -p /tmp/mi_inputs
docker cp "$ZOO_ONNX" $C:/tmp/mi_inputs/
# run_phase.sh collects the summary and raw TSVs; the extra files are written
# beside the summary inside the container and copied out here.
"$ROOT/scripts/run_phase.sh" phase0b_sam_check.py phase0b_weights || { docker exec $C rm -rf /tmp/mi_inputs; exit 1; }
docker exec $C rm -rf /tmp/mi_inputs
