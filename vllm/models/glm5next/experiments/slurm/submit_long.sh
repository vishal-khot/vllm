#!/usr/bin/env bash
# Submit experiment 5 as one 8-GPU job. Run on the cluster login node from the
# experiments worktree. Writes only $BASE/exp_long2m (new; refuses if it exists unless LONG_RESUME=1).
#   BASE=/mnt/weka/shrd/ci/haok/glm5 bash submit_long.sh
set -euo pipefail
BASE="${BASE:-/mnt/weka/shrd/ci/haok/glm5}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
LONG_OUT="${BASE}/exp_long2m"
RULER="${BASE}/ruler_2m"

[ ! -e "${LONG_OUT}" ] || [ "${LONG_RESUME:-0}" = 1 ] || { echo "ERROR: ${LONG_OUT} exists; not overwriting (LONG_RESUME=1 resumes into it)" >&2; exit 2; }
[ -x "${BASE}/venv_exp/bin/python" ] || { echo "ERROR: venv_exp missing" >&2; exit 2; }
n=$(ls "${RULER}"/*/validation_2088960.jsonl 2>/dev/null | wc -l)
[ "${n}" -eq 13 ] || { echo "ERROR: ${n}/13 RULER files in ${RULER}; run prepare_ruler_2m.sh" >&2; exit 2; }
mkdir -p "${LONG_OUT}" "${BASE}/outputs"

export GLM5_REPO="${REPO}" GLM5_VENV="${BASE}/venv_exp" LONG_OUT
export HF_HOME="${BASE}/hf_exp" UV_CACHE_DIR="${BASE}/uvcache_exp"
export GLM5_NODE_SCRATCH="${BASE}/scratch_exp"
[ -d "${BASE}/flashinfer_cubins_exp" ] && export FLASHINFER_CUBIN_DIR="${BASE}/flashinfer_cubins_exp"

cd "${BASE}"
echo "repo ${REPO} @ $(git -C "${REPO}" rev-parse --short HEAD)"
sbatch --parsable "${REPO}/vllm/models/glm5next/experiments/slurm/long.sbatch"
