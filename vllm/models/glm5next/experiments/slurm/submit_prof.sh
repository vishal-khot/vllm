#!/usr/bin/env bash
# Submit experiment 4 as one 4-GPU job. Run on the cluster login node from the
# experiments worktree. Writes only $BASE/exp_profile (new; refuses if it exists).
#   BASE=/mnt/weka/shrd/ci/haok/glm5 bash submit_prof.sh
set -euo pipefail
BASE="${BASE:-/mnt/weka/shrd/ci/haok/glm5}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
PROF_OUT="${BASE}/exp_profile"

[ ! -e "${PROF_OUT}" ] || { echo "ERROR: ${PROF_OUT} exists; not overwriting" >&2; exit 2; }
[ -x "${BASE}/venv_exp/bin/python" ] || { echo "ERROR: venv_exp missing" >&2; exit 2; }
mkdir -p "${PROF_OUT}" "${BASE}/outputs"

export GLM5_REPO="${REPO}" GLM5_VENV="${BASE}/venv_exp" PROF_OUT
export HF_HOME="${BASE}/hf_exp" UV_CACHE_DIR="${BASE}/uvcache_exp"
export GLM5_NODE_SCRATCH="${BASE}/scratch_exp"
[ -d "${BASE}/flashinfer_cubins_exp" ] && export FLASHINFER_CUBIN_DIR="${BASE}/flashinfer_cubins_exp"

cd "${BASE}"
echo "repo ${REPO} @ $(git -C "${REPO}" rev-parse --short HEAD)"
sbatch --parsable "${REPO}/vllm/models/glm5next/experiments/slurm/prof.sbatch"
