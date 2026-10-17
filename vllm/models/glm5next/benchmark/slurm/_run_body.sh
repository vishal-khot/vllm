#!/usr/bin/env bash
# Shared body for the IVF lost-attention Slurm jobs (normal + preempt).
#
# Stages GLM-5.3-Flash and the RULER jsonl to node-local NVMe, ensures a
# vLLM+fork venv, then runs lost_attention_run.py. The operating points (tasks,
# n, per-length clusters and probes) are NOT computed here -- they are read from
# the explicit ivf `sweep.cells` block in runs.yaml, so that file is the single
# transparent source of truth. The sbatch wrappers set GLM5_TP and the #SBATCH
# resource lines.
#
# Required (export before sbatch, or edit the default):
#   RULER_SRC   persistent dir holding <task>/validation_<len>.jsonl
#               (the former Modal `glm5-ruler` volume contents).
# Optional:
#   GLM5_REPO   the vllm repo root; defaults to this script's location. Set it
#               when running from a cluster where the repo is elsewhere.
#   HF_TOKEN    if zai-org/GLM-5.3-Flash is gated for you.
set -euo pipefail

# Repo root: honor GLM5_REPO, else derive from this file's location (works on
# any cluster -- this body is exec'd by absolute path, so BASH_SOURCE is real).
REPO="${GLM5_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)}"
RUNS_YAML="${REPO}/vllm/models/glm5next/benchmark/runs.yaml"
RUNNER="${REPO}/vllm/models/glm5next/benchmark/slurm/lost_attention_run.py"
MODEL_REPO="zai-org/GLM-5.3-Flash"
NODE_SCRATCH="/scratch/${USER}/glm5"
MODEL_DIR="${NODE_SCRATCH}/GLM-5.3-Flash"
VENV="${REPO}/.venv"
RULER_SRC="${RULER_SRC:-}"

: "${GLM5_TP:?set by the sbatch wrapper}"
JOB="${SLURM_JOB_ID:-local}"
OUT_ROOT="${OUT_ROOT:-${HOME}/glm5_lost_attention/${JOB}}"
NODE_OUT="${NODE_SCRATCH}/out/${JOB}"
RULER_LOCAL="${NODE_SCRATCH}/ruler"

echo "[$(date +%T)] host=$(hostname) tp=${GLM5_TP}"
nvidia-smi -L || true
mkdir -p "${NODE_SCRATCH}" "${NODE_OUT}" "${OUT_ROOT}"

export PATH="${HOME}/.local/bin:${PATH}"   # uv installs here
export HF_HOME="${NODE_SCRATCH}/hf"
export HF_HUB_ENABLE_HF_TRANSFER=1

# ---- 1. vLLM + fork venv (built once; .venv lives in the persistent repo) ----
if ! "${VENV}/bin/python" -c "import vllm" >/dev/null 2>&1; then
  echo "[$(date +%T)] building vLLM venv (VLLM_USE_PRECOMPILED) ..."
  uv venv --python 3.12 "${VENV}"
  # .git is present on Babel, so setuptools-scm resolves the version; IVF is
  # pure Python/Triton so the precompiled .so are reused (no CUDA rebuild).
  ( cd "${REPO}" && VLLM_USE_PRECOMPILED=1 \
      uv pip install --python "${VENV}/bin/python" -e . --torch-backend=auto )
fi
source "${VENV}/bin/activate"
python -c "import vllm; print('vllm', vllm.__version__)"

# ---- 2. tasks + lengths to stage, read straight from runs.yaml ----
read -r TASKS LENGTHS < <(python - "${RUNS_YAML}" <<'PY'
import sys, yaml
c = yaml.safe_load(open(sys.argv[1]))
cells = next(r["sweep"]["cells"] for r in c["runs"]
             if (r.get("sweep") or {}).get("cells"))
tasks = cells.get("tasks", "all")
if tasks in ("all", None):
    tasks = c["datasets"]["ruler"]["tasks"]
lengths = [str(g["length"]) for g in cells["grid"]]
print(",".join(tasks), ",".join(lengths))
PY
)
echo "[$(date +%T)] tasks=${TASKS} lengths=${LENGTHS}"

# ---- 3. model -> node-local NVMe (resumable; re-downloaded on a fresh node) ----
if [ ! -f "${MODEL_DIR}/config.json" ]; then
  echo "[$(date +%T)] downloading ${MODEL_REPO} -> ${MODEL_DIR}"
  df -h "${NODE_SCRATCH}" | tail -1
  python - <<PY
from huggingface_hub import snapshot_download
p = snapshot_download("${MODEL_REPO}", local_dir="${MODEL_DIR}",
                      max_workers=8, resume_download=True)
print("model ->", p)
PY
else
  echo "[$(date +%T)] model already staged at ${MODEL_DIR}"
fi

# ---- 4. RULER jsonl -> node-local ----
if [ -z "${RULER_SRC}" ]; then
  echo "ERROR: RULER_SRC is unset. Point it at the dir holding" >&2
  echo "       <task>/validation_<len>.jsonl (the old glm5-ruler volume)." >&2
  exit 2
fi
IFS=',' read -ra TASK_ARR <<< "${TASKS}"
IFS=',' read -ra LEN_ARR <<< "${LENGTHS}"
for T in "${TASK_ARR[@]}"; do
  mkdir -p "${RULER_LOCAL}/${T}"
  for L in "${LEN_ARR[@]}"; do
    src="${RULER_SRC}/${T}/validation_${L}.jsonl"
    [ -f "${src}" ] || { echo "ERROR: missing ${src}" >&2; exit 2; }
    cp -n "${src}" "${RULER_LOCAL}/${T}/" || true
  done
done

# ---- 5. run (cells, clusters, probes all come from runs.yaml) ----
echo "[$(date +%T)] running lost_attention_run.py -> ${NODE_OUT}"
python "${RUNNER}" \
  --model "${MODEL_DIR}" \
  --ruler-dir "${RULER_LOCAL}" \
  --out-dir "${NODE_OUT}" \
  --runs-yaml "${RUNS_YAML}" \
  --tp "${GLM5_TP}"

# ---- 6. copy results off node-local to persistent storage ----
echo "[$(date +%T)] copying results -> ${OUT_ROOT}"
cp -r "${NODE_OUT}/." "${OUT_ROOT}/"
echo "[$(date +%T)] done. results in ${OUT_ROOT}"
