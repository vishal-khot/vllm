#!/usr/bin/env bash
# Experiment 5: one half of the 2M samples on this step's 4 GPUs.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
: "${LONG_OUT:?}" "${LONG_HALF:?}"
set +e
python -X faulthandler "${EXP_DIR}/long_run.py" --out-dir "${LONG_OUT}" \
  --port "${GLM5_PORT}" --half "${LONG_HALF}"
rc=$?
set -e
echo "[$(date +%T)] long half ${LONG_HALF} exit ${rc} -> ${LONG_OUT}"
exit "${rc}"
