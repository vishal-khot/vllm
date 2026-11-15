#!/usr/bin/env bash
# Experiment 5: a smoke input first, then one shard per GPU, then the summary.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
: "${EMEANS_OUT:?}" "${KEYS_DIR:?}"
REF=()
[ -d "${EMEANS_OUT}/ref/faiss_emeans_ref" ] && REF=(--ref-dir "${EMEANS_OUT}/ref")
# The branch's vllm must be the one imported, not venv_exp's editable install.
python - <<PY
import vllm, vllm.models.glm5next.nvidia.ops.ivf_emeans as m
assert m.__file__.startswith("${GLM5_REPO}"), m.__file__
print("[emeans] vllm from", vllm.__file__)
PY
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
[ -f "${EMEANS_OUT}/smoke/done" ] || {
  CUDA_VISIBLE_DEVICES="${GPUS[0]}" run smoke python -X faulthandler \
    "${EXP_DIR}/emeans_bench.py" --out-dir "${EMEANS_OUT}/smoke" \
    --keys-dir "${KEYS_DIR}" --lengths 65536 --max-files 1 "${REF[@]}"
  touch "${EMEANS_OUT}/smoke/done"
}
pids=()
for i in "${!GPUS[@]}"; do
  CUDA_VISIBLE_DEVICES="${GPUS[$i]}" run "emeans g${i}" python -X faulthandler \
    "${EXP_DIR}/emeans_bench.py" --out-dir "${EMEANS_OUT}" --keys-dir "${KEYS_DIR}" \
    --shard "${i}" --num-shards "${N}" "${REF[@]}" &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "${p}" || true; done
python "${EXP_DIR}/summarize.py" --emeans "${EMEANS_OUT}"
echo "[$(date +%T)] emeans experiment complete -> ${EMEANS_OUT}"
