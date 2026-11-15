#!/usr/bin/env bash
# Submit the remaining capacity cells of experiment 1 as one 8-GPU job. Run on
# the cluster login node from the experiments worktree. Writes only those
# cells (finished cells are skipped, never rewritten) under $BASE/exp_capacity.
#   BASE=/mnt/weka/shrd/ci/haok/glm5 CAP_ONLY=ivf_C510_P128_L1040384 bash submit_cap.sh
set -euo pipefail
BASE="${BASE:-/mnt/weka/shrd/ci/haok/glm5}"
: "${CAP_ONLY:?comma list of capacity cell ids}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
CAP_OUT="${BASE}/exp_capacity"
[ -d "${CAP_OUT}" ] || { echo "ERROR: ${CAP_OUT} missing; nothing to resume" >&2; exit 2; }
[ -x "${BASE}/venv_exp/bin/python" ] || { echo "ERROR: venv_exp missing" >&2; exit 2; }
mkdir -p "${BASE}/outputs"
export GLM5_REPO="${REPO}" GLM5_VENV="${BASE}/venv_exp" CAP_OUT CAP_ONLY
export HF_HOME="${BASE}/hf_exp" UV_CACHE_DIR="${BASE}/uvcache_exp"
export GLM5_NODE_SCRATCH="${BASE}/scratch_exp"
[ -d "${BASE}/flashinfer_cubins_exp" ] && export FLASHINFER_CUBIN_DIR="${BASE}/flashinfer_cubins_exp"
cd "${BASE}"
echo "repo ${REPO} @ $(git -C "${REPO}" rev-parse --short HEAD)"
sbatch --parsable "${REPO}/vllm/models/glm5next/experiments/slurm/cap_cell.sbatch"
