#!/usr/bin/env bash
# Experiment 2 step: capture real keys with a TP=4 serve, then replay the
# microbenchmark as one process per GPU of this step.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
: "${REORDER_OUT:?}"
IFS=',' read -ra GPUS <<< "${CUDA_VISIBLE_DEVICES}"
# Smoke: one small synthetic cell, so a replay bug shows before the capture.
[ -f "${REORDER_OUT}/smoke/results/synthetic.g0.jsonl" ] || \
  CUDA_VISIBLE_DEVICES="${GPUS[0]}" python "${EXP_DIR}/reorder_bench.py" \
    --out-dir "${REORDER_OUT}/smoke" --kind synthetic --lengths 65536 --rows 128 --reps 2
python "${EXP_DIR}/capture_run.py" --out-dir "${REORDER_OUT}" --port "${GLM5_PORT}"

N=${#GPUS[@]}
for kind in real synthetic; do
  pids=()
  for i in "${!GPUS[@]}"; do
    CUDA_VISIBLE_DEVICES="${GPUS[$i]}" python "${EXP_DIR}/reorder_bench.py" \
      --out-dir "${REORDER_OUT}" --kind "${kind}" --shard "${i}" --num-shards "${N}" &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "${p}"; done
done
python "${EXP_DIR}/summarize.py" --reorder "${REORDER_OUT}"
echo "[$(date +%T)] reorder experiment complete -> ${REORDER_OUT}"
