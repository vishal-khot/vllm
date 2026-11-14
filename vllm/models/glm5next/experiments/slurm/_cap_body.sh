#!/usr/bin/env bash
# Experiment 1 step: capacity bisection on this step's 4 GPUs.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
: "${CAP_OUT:?}"
python "${EXP_DIR}/capacity_run.py" --out-dir "${CAP_OUT}" --port "${GLM5_PORT}"
