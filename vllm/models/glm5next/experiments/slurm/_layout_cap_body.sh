#!/usr/bin/env bash
# Experiment 3 on each GPU, then the capacity cells left from job 3055563.
# Each runner's exit code is logged (3055563's capacity step exited 1 silently).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
: "${CAP_OUT:?}" "${LAYOUT_OUT:?}" "${KEYS_DIR:?}"
IFS=',' read -ra GPUS <<< "${CUDA_VISIBLE_DEVICES}"
N=${#GPUS[@]}
run() {  # label cmd...
  set +e
  "${@:2}"
  local rc=$?
  set -e
  echo "[$(date +%T)] $1 exit ${rc}"
  return "${rc}"
}

# Smoke: one small input through every part, so a kernel bug shows first.
[ -f "${LAYOUT_OUT}/smoke/done" ] || {
  CUDA_VISIBLE_DEVICES="${GPUS[0]}" run smoke python -X faulthandler \
    "${EXP_DIR}/layout_bench.py" --out-dir "${LAYOUT_OUT}/smoke" \
    --keys-dir "${KEYS_DIR}" --lengths 65536 --max-files 1 --rows 128,1024 --reps 2
  touch "${LAYOUT_OUT}/smoke/done"
}

pids=()
for i in "${!GPUS[@]}"; do
  CUDA_VISIBLE_DEVICES="${GPUS[$i]}" run "layout g${i}" python -X faulthandler \
    "${EXP_DIR}/layout_bench.py" --out-dir "${LAYOUT_OUT}" --keys-dir "${KEYS_DIR}" \
    --shard "${i}" --num-shards "${N}" &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "${p}" || true; done  # capacity runs regardless
python "${EXP_DIR}/summarize.py" --layout "${LAYOUT_OUT}"
echo "[$(date +%T)] layout experiment complete -> ${LAYOUT_OUT}"

run capacity python -X faulthandler "${EXP_DIR}/capacity_run.py" \
  --out-dir "${CAP_OUT}" --port "${GLM5_PORT}"
echo "[$(date +%T)] capacity complete -> ${CAP_OUT}"
