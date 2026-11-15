#!/usr/bin/env bash
# Experiment 4: profile one served prefill per cell; the exit code is logged.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
: "${PROF_OUT:?}"
set +e
python -X faulthandler "${EXP_DIR}/prefill_profile.py" --out-dir "${PROF_OUT}" \
  --port "${GLM5_PORT}"
rc=$?
set -e
echo "[$(date +%T)] profile exit ${rc} -> ${PROF_OUT}"
exit "${rc}"
