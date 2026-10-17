#!/usr/bin/env bash
# One-shot environment setup for the GLM-5.3-Flash DSA-indexer benchmark.
#
# Run this ONCE on the target cluster, then submit the sbatch jobs. It installs
# uv, builds the vLLM+fork venv (if a GPU is visible), and sanity-checks the
# RULER data. The ~306 GB model download is NOT done here -- it happens on the
# first Slurm job, to node-local scratch.
#
# Usage:
#   ./setup.sh [RULER_SRC]
#   RULER_SRC=/path/to/ruler ./setup.sh
#
# The venv build needs CUDA visible to select the right torch backend. On a
# GPU-less login node this script defers the build to the first Slurm job (which
# runs on a GPU node and builds it automatically). To build now instead, run
# this script inside a GPU allocation (see the hint it prints).
set -euo pipefail

REPO="${GLM5_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)}"
VENV="${REPO}/.venv"
RULER_SRC="${1:-${RULER_SRC:-}}"
SLURM_DIR="${REPO}/vllm/models/glm5next/benchmark/slurm"

say() { printf '\n[setup] %s\n' "$*"; }

say "repo root: ${REPO}"

# ---- 1. uv ----
export PATH="${HOME}/.local/bin:${PATH}"
if ! command -v uv >/dev/null 2>&1; then
  say "installing uv -> ~/.local/bin ..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
fi
command -v uv >/dev/null 2>&1 || {
  echo "[setup] ERROR: uv still not on PATH; add ~/.local/bin to PATH" >&2
  exit 1
}
say "uv: $(uv --version)"

# ---- 2. venv + vLLM (fork) ----
if "${VENV}/bin/python" -c "import vllm" >/dev/null 2>&1; then
  say "venv ready: vllm $("${VENV}/bin/python" -c 'import vllm; print(vllm.__version__)')"
elif command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  say "GPU visible -> building venv (VLLM_USE_PRECOMPILED, no CUDA rebuild) ..."
  uv venv --python 3.12 "${VENV}"
  ( cd "${REPO}" && VLLM_USE_PRECOMPILED=1 \
      uv pip install --python "${VENV}/bin/python" -e . --torch-backend=auto )
  "${VENV}/bin/python" -c "import vllm; print('[setup] vllm', vllm.__version__)"
else
  say "no GPU on this node -> deferring the vLLM build to the first Slurm job"
  say "(its body builds ${VENV} automatically on a GPU node). To build NOW, rerun"
  say "this script inside a GPU alloc, e.g.:"
  echo "    srun --partition=<p> --qos=<q> --gres=gpu:<type>:1 --pty bash ${SLURM_DIR}/setup.sh"
fi

# ---- 3. RULER data sanity check ----
if [ -n "${RULER_SRC}" ]; then
  say "checking RULER_SRC=${RULER_SRC}"
  miss=0
  for probe in niah_single_1/validation_16384.jsonl vt/validation_131072.jsonl \
               qa_2/validation_8192.jsonl; do
    if [ ! -f "${RULER_SRC}/${probe}" ]; then
      echo "   MISSING: ${RULER_SRC}/${probe}"
      miss=1
    fi
  done
  if [ "${miss}" = 0 ]; then
    say "RULER layout looks OK (<task>/validation_<len>.jsonl)"
  else
    say "RULER data looks incomplete -- fix it before submitting jobs"
  fi
else
  say "no RULER_SRC given. You MUST pass RULER_SRC=... when you sbatch."
fi

# ---- 4. next steps ----
say "setup complete. Submit from ${SLURM_DIR} (override partition/qos/gres for"
say "your cluster; set HF_TOKEN only if the model is gated):"
cat <<EOF

  cd ${SLURM_DIR}

  # cheap smoke first (1 run, 1 arm):
  GLM5_ONLY=ivf RULER_SRC=<path> sbatch \\
      --partition=<p> --qos=<q> --gres=gpu:<type>:4 matrix_preempt.sbatch

  # full A/B matrix (26 runs):
  RULER_SRC=<path> sbatch \\
      --partition=<p> --qos=<q> --gres=gpu:<type>:8 matrix_normal.sbatch

  # IVF fidelity sweep (25 cells):
  RULER_SRC=<path> sbatch \\
      --partition=<p> --qos=<q> --gres=gpu:<type>:8 lost_attention_normal.sbatch

Results land in \$HOME/glm5_ab_matrix/<jobid>/ and
\$HOME/glm5_lost_attention/<jobid>/ (set OUT_ROOT=... to change).
EOF
