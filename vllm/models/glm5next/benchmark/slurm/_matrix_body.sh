#!/usr/bin/env bash
# Shared body for the A/B matrix Slurm jobs (normal + preempt).
#
# Stages GLM-5.3-Flash and the RULER jsonl to node-local NVMe, ensures a
# vLLM+fork venv, then runs slurm/matrix_run.py over every run in runs.yaml.
# The run matrix (indexer arms, lengths, clusters/probes, concurrency) is read
# from runs.yaml via bench_config -- nothing is computed here. The sbatch
# wrappers set GLM5_TP and the #SBATCH resource lines.
#
# Required (export before sbatch, or edit the default):
#   RULER_SRC   persistent dir holding <task>/validation_<len>.jsonl
#               (the former Modal `glm5-ruler` volume contents).
# Optional:
#   GLM5_REPO   the vllm repo root; defaults to this script's location. Set it
#               when running from a cluster where the repo is elsewhere.
#   GLM5_ONLY   comma list of run ids to restrict to (default: all).
#   HF_TOKEN    if zai-org/GLM-5.3-Flash is gated for you.
set -euo pipefail

# Repo root: honor GLM5_REPO, else derive from this file's location.
REPO="${GLM5_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)}"
RUNS_YAML="${REPO}/vllm/models/glm5next/benchmark/runs.yaml"
RUNNER="${REPO}/vllm/models/glm5next/benchmark/slurm/matrix_run.py"
MODEL_REPO="zai-org/GLM-5.3-Flash"
NODE_SCRATCH="/scratch/${USER}/glm5"
MODEL_DIR="${NODE_SCRATCH}/GLM-5.3-Flash"
VENV="${GLM5_VENV:-${REPO}/.venv}"
RULER_SRC="${RULER_SRC:-}"

: "${GLM5_TP:?set by the sbatch wrapper}"
JOB="${SLURM_JOB_ID:-local}"
# Results are written straight to persistent OUT_ROOT (not node-local scratch):
# SLURM_JOB_ID survives --requeue, so after preemption the runner resumes into
# the same dir and skips the runs that already wrote verdict.json.
OUT_ROOT="${OUT_ROOT:-${HOME}/glm5_ab_matrix/${JOB}}"
RULER_LOCAL="${NODE_SCRATCH}/ruler"

# Shared persistent dirs from run_benchmark.sh take precedence over node-local
# staging: serve straight from them, no per-node download/copy.
[ -n "${SHARED_MODEL_DIR:-}" ] && MODEL_DIR="${SHARED_MODEL_DIR}"
[ -n "${SHARED_RULER_DIR:-}" ] && RULER_LOCAL="${SHARED_RULER_DIR}"

echo "[$(date +%T)] host=$(hostname) tp=${GLM5_TP} repo=${REPO}"
nvidia-smi -L || true
mkdir -p "${NODE_SCRATCH}" "${OUT_ROOT}"

export PATH="${HOME}/.local/bin:${PATH}"   # uv installs here
export HF_HOME="${NODE_SCRATCH}/hf"
export HF_HUB_ENABLE_HF_TRANSFER=1

# ---- 1. vLLM + fork venv (built once; .venv lives in the persistent repo) ----
if ! "${VENV}/bin/python" -c "import vllm" >/dev/null 2>&1; then
  echo "[$(date +%T)] building vLLM venv (VLLM_USE_PRECOMPILED) ..."
  uv venv --python 3.12 "${VENV}"
  # .git present -> setuptools-scm resolves the version; IVF is pure
  # Python/Triton so the precompiled .so are reused (no CUDA rebuild).
  ( cd "${REPO}" && VLLM_USE_PRECOMPILED=1 \
      uv pip install --python "${VENV}/bin/python" -e . \
        --torch-backend="${TORCH_BACKEND:-auto}" )
fi
source "${VENV}/bin/activate"
python -c "import vllm; print('vllm', vllm.__version__)"

# ---- 2. tasks + lengths to stage: union over every ruler run in the matrix ----
read -r TASKS LENGTHS < <(python - "${RUNS_YAML}" "${REPO}" <<'PY'
import sys
sys.path.insert(0, sys.argv[2] + "/vllm/models/glm5next/benchmark")
from bench_config import load_matrix
m = load_matrix(sys.argv[1])
tasks, lengths = set(), set()
for r in m.runs:
    for d in r.datasets:
        if d.name == "ruler":
            tasks.update(d.tasks)
            lengths.update(d.lengths)
print(",".join(sorted(tasks)), ",".join(str(x) for x in sorted(lengths)))
PY
)
echo "[$(date +%T)] tasks=${TASKS} lengths=${LENGTHS}"

# ---- 3. model (shared dir if given, else node-local NVMe; resumable) ----
if [ -n "${SHARED_MODEL_DIR:-}" ]; then
  echo "[$(date +%T)] using shared model ${MODEL_DIR}"
  [ -f "${MODEL_DIR}/config.json" ] || { echo "ERROR: no config.json in ${MODEL_DIR}" >&2; exit 2; }
elif [ ! -f "${MODEL_DIR}/config.json" ]; then
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

# ---- 4. RULER jsonl (shared dir if given, else stage from RULER_SRC) ----
if [ -n "${SHARED_RULER_DIR:-}" ]; then
  echo "[$(date +%T)] using shared ruler ${RULER_LOCAL}"
else
  [ -n "${RULER_SRC}" ] || {
    echo "ERROR: set RULER_SRC (dir of <task>/validation_<len>.jsonl) or" >&2
    echo "       SHARED_RULER_DIR (run_benchmark.sh does this)." >&2
    exit 2
  }
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
fi

# ---- 5. run the matrix (arms, lengths, clusters/probes all from runs.yaml) ----
# Resume-safe: runs that already wrote verdict.json in OUT_ROOT are skipped.
echo "[$(date +%T)] running matrix_run.py -> ${OUT_ROOT}"
python "${RUNNER}" \
  --model "${MODEL_DIR}" \
  --ruler-dir "${RULER_LOCAL}" \
  --out-dir "${OUT_ROOT}" \
  --runs-yaml "${RUNS_YAML}" \
  --tp "${GLM5_TP}" \
  ${GLM5_ONLY:+--only "${GLM5_ONLY}"}
echo "[$(date +%T)] done. results in ${OUT_ROOT}"
