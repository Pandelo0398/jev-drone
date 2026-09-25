#!/usr/bin/env bash
# Heuristic baseline vs Laya onboard, same seeds, same perception/physics/controller.
#   bench/run_benchmark.sh [out_dir] [seeds...]
# Needs the onboard service up:  docker compose -f docker-compose.onboard.yml up -d
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${1:-runs/bench}; shift || true
SEEDS=${*:-0 1 2 3 4 5 6 7 8 9}
export MUJOCO_GL=${MUJOCO_GL:-egl}
mkdir -p "$OUT"
# GPU memory with the model NOT loaded, so the model's share can be reported
docker stop laya-onboard >/dev/null
sleep 3
IDLE=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
docker start laya-onboard >/dev/null
until curl -sf localhost:8000/healthz >/dev/null; do sleep 2; done
echo "gpu idle ${IDLE} MiB" | tee "$OUT/gpu_idle.txt"

.venv/bin/python run.py --no-jev --fast --seconds 65 --seeds $SEEDS 2>/dev/null \
  | grep '^{' > "$OUT/heuristic.jsonl"
.venv/bin/python run.py --decision-provider laya --edge-profile jetson_orin_nano_like \
  --seconds 65 --seeds $SEEDS --log-dir "$OUT/decisions" --gpu-idle-mb "$IDLE" 2>/dev/null \
  | grep '^{' > "$OUT/laya.jsonl"
.venv/bin/python bench/summarize.py "$OUT"
