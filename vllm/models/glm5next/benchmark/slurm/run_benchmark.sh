#!/usr/bin/env bash
# ONE command to run the whole GLM-5.3-Flash RULER benchmark on a Slurm cluster.
# It does everything:
#   1. ensure the fork is cloned and on the `benchmarking` branch,
#   2. install uv,
#   3. download the model to ./model (current dir),
#   4. generate RULER data to ./ruler (current dir),
#   5. build the vLLM venv once (a short GPU job),
#   6. submit the A/B matrix and the IVF fidelity sweep (both wait for the build).
#
# Usage (set your cluster's partition/qos/gres; #GPUs in GRES becomes TP):
#   PARTITION=<p> QOS=<q> GRES=gpu:<type>:8 bash run_benchmark.sh
#
# Required : PARTITION, QOS, GRES (e.g. gpu:A100_80GB:8).
# Optional : HF_TOKEN (gated model), MODE=normal|preempt (default normal),
#            MODEL_DIR (default ./model), RULER_DIR (default ./ruler),
#            NUM_SAMPLES (RULER samples/task, default 100), GLM5_ONLY=<run ids>,
#            REPO_URL, BRANCH (default benchmarking), CLONE_DEST.
#
# Tip: the model download is large; run under tmux/nohup so a dropped session
# does not kill it.
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/vishal-khot/vllm.git}"
BRANCH="${BRANCH:-benchmarking}"
MODE="${MODE:-normal}"
NUM_SAMPLES="${NUM_SAMPLES:-100}"
MODEL_DIR="${MODEL_DIR:-${PWD}/model}"
RULER_DIR="${RULER_DIR:-${PWD}/ruler}"
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

say() { printf '\n[run] %s\n' "$*"; }

# ---- required cluster inputs ----
if [ -z "${PARTITION:-}" ] || [ -z "${QOS:-}" ] || [ -z "${GRES:-}" ]; then
  echo "ERROR: set PARTITION, QOS and GRES. Example:" >&2
  echo "  PARTITION=general QOS=normal GRES=gpu:A100_80GB:8 bash run_benchmark.sh" >&2
  echo "Partitions/GRES available here:" >&2
  sinfo -o "  %P %l %G" 2>/dev/null | sort -u >&2 || true
  exit 2
fi
TP="${GRES##*:}"
[[ "${TP}" =~ ^[0-9]+$ ]] || { echo "ERROR: GRES must end in :<N> (e.g. gpu:A100_80GB:8)" >&2; exit 2; }
case "${MODE}" in normal|preempt) ;; *) echo "ERROR: MODE must be normal|preempt" >&2; exit 2 ;; esac

# ---- 1. repo on the right branch (clone if absent) ----
if [ -n "${GLM5_REPO:-}" ]; then
  REPO="${GLM5_REPO}"
elif [ -f "${SELF_DIR}/../runs.yaml" ]; then
  REPO="$(cd "${SELF_DIR}/../../../../.." && pwd)"        # script lives in the repo
else
  REPO="${CLONE_DEST:-${PWD}/vllm}"
  [ -d "${REPO}/.git" ] || { say "cloning ${REPO_URL}"; git clone "${REPO_URL}" "${REPO}"; }
fi
say "repo: ${REPO}"
( cd "${REPO}" && git fetch origin "${BRANCH}" && git checkout "${BRANCH}" \
    && git pull --ff-only origin "${BRANCH}" ) \
  || say "WARNING: could not sync ${BRANCH} (continuing with the checked-out tree)"
SLURM_DIR="${REPO}/vllm/models/glm5next/benchmark/slurm"

# ---- 2. uv ----
export PATH="${HOME}/.local/bin:${PATH}"
if ! command -v uv >/dev/null 2>&1; then
  say "installing uv -> ~/.local/bin"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
fi
say "uv: $(uv --version)"

# ---- 3. model -> ./model ----
mkdir -p "${MODEL_DIR}"
if [ -f "${MODEL_DIR}/config.json" ]; then
  say "model already present at ${MODEL_DIR}"
else
  say "downloading zai-org/GLM-5.3-Flash -> ${MODEL_DIR} (large; resumable)"
  HF_HUB_ENABLE_HF_TRANSFER=1 uv run --with huggingface_hub --with hf_transfer \
    python - "${MODEL_DIR}" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download("zai-org/GLM-5.3-Flash", local_dir=sys.argv[1],
                  max_workers=8, resume_download=True)
print("model ->", sys.argv[1])
PY
fi

# ---- 4. RULER -> ./ruler (skip if already populated or RULER_DIR supplied) ----
if [ -f "${RULER_DIR}/niah_single_1/validation_16384.jsonl" ]; then
  say "RULER data already present at ${RULER_DIR}"
else
  say "generating RULER -> ${RULER_DIR} (official pipeline, GLM tokenizer)"
  MODEL_DIR="${MODEL_DIR}" OUT="${RULER_DIR}" NUM_SAMPLES="${NUM_SAMPLES}" \
    bash "${SLURM_DIR}/prepare_ruler.sh"
fi
[ -f "${RULER_DIR}/niah_single_1/validation_16384.jsonl" ] || {
  echo "ERROR: RULER data missing in ${RULER_DIR}; cannot submit jobs." >&2
  echo "       Fix generation, or set RULER_DIR to a pre-generated dir." >&2
  exit 2
}

# Pass config to the jobs through the environment (carried by --export=ALL).
# NOT via --export=k=v,... because GLM5_ONLY may contain commas, which SLURM
# would misparse as extra assignments.
export GLM5_REPO="${REPO}" GLM5_TP="${TP}"
export SHARED_MODEL_DIR="${MODEL_DIR}" SHARED_RULER_DIR="${RULER_DIR}"
[ -n "${HF_TOKEN:-}" ] && export HF_TOKEN
[ -n "${GLM5_ONLY:-}" ] && export GLM5_ONLY

# ---- 5. build the venv once, on a GPU node (1 GPU) ----
BUILD_GRES="${GRES%:*}:1"
BUILD_ID=$(sbatch --parsable --job-name=glm5-build \
  --partition="${PARTITION}" --qos="${QOS}" --gres="${BUILD_GRES}" \
  --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G --time=2:00:00 \
  --export=ALL --output="glm5-build-%j.out" \
  --wrap="bash ${SLURM_DIR}/_build_venv.sh")
say "build job: ${BUILD_ID}"

# ---- 6. submit matrix + fidelity, both after the build ----
common=(--partition="${PARTITION}" --qos="${QOS}" --gres="${GRES}"
        --dependency="afterok:${BUILD_ID}" --export=ALL)
M_ID=$(sbatch --parsable "${common[@]}" "${SLURM_DIR}/matrix_${MODE}.sbatch")
F_ID=$(sbatch --parsable "${common[@]}" "${SLURM_DIR}/lost_attention_${MODE}.sbatch")

say "submitted:"
echo "    build    ${BUILD_ID}"
echo "    matrix   ${M_ID}  -> \$HOME/glm5_ab_matrix/${M_ID}/"
echo "    fidelity ${F_ID}  -> \$HOME/glm5_lost_attention/${F_ID}/"
echo "  watch: squeue -u \$USER"
