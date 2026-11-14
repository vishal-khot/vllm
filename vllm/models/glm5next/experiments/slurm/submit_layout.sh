#!/usr/bin/env bash
# Submit experiment 3 plus the capacity cells left from job 3055563 as one
# 4-GPU job. Run on the cluster login node from the experiments worktree.
# Writes only $BASE/exp_layout (new; refuses if it exists) and the missing
# cells of $BASE/exp_capacity (finished cells are skipped, never rewritten);
# reads experiment 2's captured keys.
#   BASE=/mnt/weka/shrd/ci/haok/glm5 bash submit_layout.sh
set -euo pipefail
BASE="${BASE:-/mnt/weka/shrd/ci/haok/glm5}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
CAP_OUT="${BASE}/exp_capacity"
LAYOUT_OUT="${BASE}/exp_layout"
KEYS_DIR="${BASE}/exp_reorder/keys"

[ ! -e "${LAYOUT_OUT}" ] || { echo "ERROR: ${LAYOUT_OUT} exists; not overwriting" >&2; exit 2; }
[ -d "${CAP_OUT}" ] || { echo "ERROR: ${CAP_OUT} missing; nothing to resume" >&2; exit 2; }
[ -d "${KEYS_DIR}" ] || { echo "ERROR: ${KEYS_DIR} missing" >&2; exit 2; }
[ -x "${BASE}/venv_exp/bin/python" ] || { echo "ERROR: venv_exp missing" >&2; exit 2; }
mkdir -p "${LAYOUT_OUT}/smoke" "${BASE}/outputs"

export GLM5_REPO="${REPO}" GLM5_VENV="${BASE}/venv_exp"
export CAP_OUT LAYOUT_OUT KEYS_DIR
export HF_HOME="${BASE}/hf_exp" UV_CACHE_DIR="${BASE}/uvcache_exp"
export GLM5_NODE_SCRATCH="${BASE}/scratch_exp"
[ -d "${BASE}/flashinfer_cubins_exp" ] && export FLASHINFER_CUBIN_DIR="${BASE}/flashinfer_cubins_exp"

cd "${BASE}"
echo "repo ${REPO} @ $(git -C "${REPO}" rev-parse --short HEAD)"
sbatch --parsable "${REPO}/vllm/models/glm5next/experiments/slurm/layout_cap.sbatch"
