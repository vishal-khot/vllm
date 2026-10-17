#!/usr/bin/env bash
# Build the vLLM+fork venv once, on a GPU node (so --torch-backend=auto picks
# the CUDA wheel). Idempotent: skips if vllm already imports. Run as its own
# short Slurm job by run_benchmark.sh, so the matrix/fidelity jobs that depend
# on it never race to build the shared $REPO/.venv.
set -euo pipefail
REPO="${GLM5_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)}"
VENV="${REPO}/.venv"
export PATH="${HOME}/.local/bin:${PATH}"

if "${VENV}/bin/python" -c "import vllm" >/dev/null 2>&1; then
  echo "[build] venv ready: vllm $("${VENV}/bin/python" -c 'import vllm; print(vllm.__version__)')"
  exit 0
fi
echo "[build] building ${VENV} (VLLM_USE_PRECOMPILED; IVF is pure Python/Triton)"
uv venv --python 3.12 "${VENV}"
( cd "${REPO}" && VLLM_USE_PRECOMPILED=1 \
    uv pip install --python "${VENV}/bin/python" -e . --torch-backend=auto )
"${VENV}/bin/python" -c "import vllm; print('[build] vllm', vllm.__version__)"
