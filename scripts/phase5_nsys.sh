#!/bin/bash
# Phase 5 cross-check: GPU kernel time of ONE search call (256 crop, deployable
# setting), from an Nsight Systems timeline (no kernel replay, normal clocks).
# Nsight Compute's summed durations ran 1.2-4x above the timed latencies; its
# capture included the warm-up calls for FAISS and cuVS (4x the kernel count seen
# here), so phase5_ncu.tsv is used for its throughput percentages only, and this
# is the per-crop GPU time.
# Usage: scripts/phase5_nsys.sh   Output: results/phase5_nsys.tsv
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/results/phase5_nsys.tsv"
"$ROOT/scripts/gpu_lock.sh" acquire inspection phase5_nsys
trap '"$ROOT/scripts/gpu_lock.sh" release inspection' EXIT
printf "bank\tmethod\tkernels\tkernel_ms\n" > "$OUT"
for b in 100000 1000000; do
  for m in trt_bf faiss_flat16 faiss_ivf faiss_ivfpq cagra; do
    docker exec -e NCU=1 mi-search bash -c "cd /tmp && nsys profile -f true -o /tmp/ns_${m}_$b \
      --capture-range=cudaProfilerApi --capture-range-end=stop -t cuda \
      python3 /mi/scripts/phase5_worker.py $m $b 1 >/dev/null 2>&1; \
      nsys stats -r cuda_gpu_kern_sum -f csv /tmp/ns_${m}_$b.nsys-rep 2>/dev/null" \
    | grep -E "^(Time|[0-9])" | python3 -c "
import csv, sys
r = list(csv.DictReader(sys.stdin))
print('$b\t$m\t%d\t%.3f' % (sum(int(x['Instances']) for x in r), sum(float(x['Total Time (ns)']) for x in r) / 1e6))" >> "$OUT"
  done
done
cat "$OUT"
