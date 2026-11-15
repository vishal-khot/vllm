#!/usr/bin/env bash
# Submit experiment 5 (2 GPUs). Run on the cluster login node from any
# checkout; it builds its own worktree and never touches another:
#   BASE=/mnt/weka/shrd/ci/haok/glm5 SHA=<emeans-v0.32 commit> bash submit_emeans.sh
# - worktree $WT (new) at $SHA, from the repo of $SRC_WT, with $SRC_WT's
#   untracked build artifacts (compiled extensions) copied in;
# - the FAISS E-means torch reference at a pinned commit, for the check;
# - output $BASE/exp_emeans (new; refuses if it exists).
set -euo pipefail
BASE="${BASE:-/mnt/weka/shrd/ci/haok/glm5}"
: "${SHA:?set SHA to the emeans-v0.32 commit}"
SRC_WT="${SRC_WT:-${HOME}/vllm-kmeans-dsa/vllm-exp}"
WT="${WT:-${HOME}/vllm-kmeans-dsa/vllm-emeans}"
EMEANS_OUT="${BASE}/exp_emeans"
KEYS_DIR="${BASE}/exp_reorder/keys"
FAISS_SHA=2b648f0605f2ef3d14c4705350f8a0b214463878
[ ! -e "${EMEANS_OUT}" ] || { echo "ERROR: ${EMEANS_OUT} exists; not overwriting" >&2; exit 2; }
[ ! -e "${WT}" ] || { echo "ERROR: ${WT} exists" >&2; exit 2; }
[ -d "${KEYS_DIR}" ] || { echo "ERROR: ${KEYS_DIR} missing" >&2; exit 2; }
[ -x "${BASE}/venv_exp/bin/python" ] || { echo "ERROR: venv_exp missing" >&2; exit 2; }

git -C "${SRC_WT}" fetch -q origin emeans-v0.32
git -C "${SRC_WT}" worktree add -q --detach "${WT}" "${SHA}"
( cd "${SRC_WT}" && git ls-files --others --ignored --exclude-standard -z -- vllm \
    | grep -zvE '__pycache__|\.pyc$' | xargs -0 -r cp --parents -t "${WT}" )
mkdir -p "${EMEANS_OUT}/ref/faiss_emeans_ref" "${EMEANS_OUT}/smoke" "${BASE}/outputs"
touch "${EMEANS_OUT}/ref/faiss_emeans_ref/__init__.py"
for f in base.py e_means_torch.py torch_backend.py e_means_spherical.py; do
  curl -sfL --max-time 60 -o "${EMEANS_OUT}/ref/faiss_emeans_ref/${f}" \
    "https://raw.githubusercontent.com/facebookresearch/faiss/${FAISS_SHA}/contrib/e_means/${f}"
done

export GLM5_REPO="${WT}" GLM5_VENV="${BASE}/venv_exp" EMEANS_OUT KEYS_DIR
export PYTHONPATH="${WT}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME="${BASE}/hf_exp" UV_CACHE_DIR="${BASE}/uvcache_exp"
cd "${BASE}"
echo "worktree ${WT} @ $(git -C "${WT}" rev-parse --short HEAD)"
sbatch --parsable "${WT}/vllm/models/glm5next/experiments/slurm/emeans.sbatch"
