#!/usr/bin/env bash
# Submit both experiments as one 8-GPU job. Run on the cluster login node from
# the experiments worktree. Everything it writes is its own:
#   $BASE/exp_capacity  $BASE/exp_reorder        results
#   $BASE/venv_exp  $BASE/hf_exp  $BASE/uvcache_exp  $BASE/scratch_exp
# and only reads the shared model. It refuses to start over existing result
# dirs, so it can't overwrite anything; a requeue of the job resumes in place.
#   BASE=/mnt/weka/shrd/ci/haok/glm5 bash submit.sh
set -euo pipefail
BASE="${BASE:-/mnt/weka/shrd/ci/haok/glm5}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
CAP_OUT="${BASE}/exp_capacity"
REORDER_OUT="${BASE}/exp_reorder"

[ -f "${REPO}/vllm/models/glm5next/experiments/runs.yaml" ] || {
  echo "ERROR: ${REPO} has no experiments/runs.yaml" >&2; exit 2; }
for d in "${CAP_OUT}" "${REORDER_OUT}"; do
  [ ! -e "${d}" ] || { echo "ERROR: ${d} exists; not overwriting" >&2; exit 2; }
done
mkdir -p "${CAP_OUT}" "${REORDER_OUT}" "${BASE}/hf_exp" "${BASE}/uvcache_exp" \
  "${BASE}/scratch_exp" "${BASE}/outputs"

export GLM5_REPO="${REPO}" GLM5_VENV="${BASE}/venv_exp"
export CAP_OUT REORDER_OUT
export HF_HOME="${BASE}/hf_exp" UV_CACHE_DIR="${BASE}/uvcache_exp"
export GLM5_NODE_SCRATCH="${BASE}/scratch_exp"
[ -d "${BASE}/flashinfer_cubins_exp" ] && export FLASHINFER_CUBIN_DIR="${BASE}/flashinfer_cubins_exp"

cd "${BASE}"
echo "repo ${REPO} @ $(git -C "${REPO}" rev-parse --short HEAD)"
sbatch --parsable "${REPO}/vllm/models/glm5next/experiments/slurm/exp_pair.sbatch"
