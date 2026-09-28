#!/bin/bash
# Phase 2 setup: engines, Triton model repository, frames, the mi-triton
# server and the client binary.
#
# mi-triton is a SECOND Triton server, separate from the companion repo's
# triton-server: its own model repository (triton/models in THIS repo), its own
# ports (HTTP 8100, gRPC 8101, metrics 8102). Nothing is added to the companion
# repo's model repository. Every variant a sweep needs is loaded at once, so no
# run restarts the server:
#   yolov8s, yolov8s_prio      stage 1 (the companion repo's engine, copied)
#   s2, s2_db0, s2_prio        stage 2 (no batcher / dynamic batcher, 0 us / PRIORITY_MAX)
#   s3_crop, s3_frame          stage 3 at 384 / 800
#
# Usage: scripts/phase2_setup.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
COMPANION=/home/suchi/sadbodh/rt_vs_triton
MR="$ROOT/triton/models"
C=triton-server
W=/tmp/mi_phase2_build

"$ROOT/scripts/gpu_lock.sh" acquire inspection phase2_setup
trap '"$ROOT/scripts/gpu_lock.sh" release inspection; docker exec $C rm -rf $W || true' EXIT

# 1) engines (stage 2 and 3), built in triton-server with Triton's own TensorRT
docker exec $C rm -rf $W
docker exec $C mkdir -p $W
for f in "$ROOT"/scripts/*.py; do docker cp "$f" $C:$W/; done
docker exec $C python3 $W/phase2_export.py $W/out
mkdir -p "$ROOT/triton/engines"
for e in s2 s3_crop s3_frame; do docker cp $C:$W/out/$e.plan "$ROOT/triton/engines/"; done

# 2) model repository
rm -rf "$MR"; mkdir -p "$MR"
put() {  # name engine config
  mkdir -p "$MR/$1/1"
  cp "$2" "$MR/$1/1/model.plan"
  printf '%s\n' "name: \"$1\"" "platform: \"tensorrt_plan\"" "$3" > "$MR/$1/config.pbtxt"
}
Y="$COMPANION/triton/models/yolov8s/1/model.plan"
PRIO='optimization { priority: PRIORITY_MAX }'
put yolov8s      "$Y" 'max_batch_size: 0
instance_group [ { count: 2 kind: KIND_GPU } ]'
put yolov8s_prio "$Y" "max_batch_size: 0
instance_group [ { count: 2 kind: KIND_GPU } ]
$PRIO"
S2="$ROOT/triton/engines/s2.plan"
put s2       "$S2" 'max_batch_size: 16
instance_group [ { count: 1 kind: KIND_GPU } ]'
put s2_db0   "$S2" 'max_batch_size: 16
instance_group [ { count: 1 kind: KIND_GPU } ]
dynamic_batching { max_queue_delay_microseconds: 0 }'
put s2_prio  "$S2" "max_batch_size: 16
instance_group [ { count: 1 kind: KIND_GPU } ]
$PRIO"
put s3_crop  "$ROOT/triton/engines/s3_crop.plan"  'max_batch_size: 0
instance_group [ { count: 1 kind: KIND_GPU } ]'
put s3_frame "$ROOT/triton/engines/s3_frame.plan" 'max_batch_size: 0
instance_group [ { count: 1 kind: KIND_GPU } ]'

# 3) the same preprocessed frames the companion repo's paced clients replay
mkdir -p "$ROOT/data"
[ -f "$ROOT/data/frames.bin" ] || docker cp $C:/work/cpp/build/frames.bin "$ROOT/data/frames.bin"

# 4) the server
docker rm -f mi-triton >/dev/null 2>&1 || true
docker run -d --name mi-triton --gpus all --network host --shm-size=1g \
  -v "$MR:/models" -v "$ROOT:/mi" triton-bench:v3 \
  tritonserver --model-repository=/models --http-port=8100 --grpc-port=8101 \
  --metrics-port=8102 --log-verbose=1 >/dev/null
for i in $(seq 1 60); do
  curl -sf localhost:8100/v2/health/ready >/dev/null && break; sleep 2
done
curl -sf localhost:8100/v2/health/ready >/dev/null || { docker logs --tail 40 mi-triton; exit 1; }
for m in yolov8s yolov8s_prio s2 s2_db0 s2_prio s3_crop s3_frame; do
  printf '%-14s %s\n' $m "$(curl -s -o /dev/null -w %{http_code} localhost:8100/v2/models/$m/ready)"
done

# 5) the client
docker exec mi-triton bash -c "mkdir -p /mi/cpp/build && cd /mi/cpp/build && cmake .. -DCMAKE_BUILD_TYPE=Release >/dev/null && make -j8 2>&1 | tail -3"
docker exec mi-triton ls -la /mi/cpp/build/inspect_client
# files the containers wrote into the repo are root-owned; hand them back
docker run --rm -v "$ROOT:/x" alpine:3 chown -R "$(id -u):$(id -g)" /x/cpp /x/data /x/triton
