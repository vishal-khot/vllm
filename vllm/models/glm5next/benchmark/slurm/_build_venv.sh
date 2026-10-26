#!/usr/bin/env bash
# Build both venvs once, on a GPU node (so --torch-backend=auto picks the CUDA
# wheel). Idempotent: skips a venv whose vllm already imports. Run as its own
# short Slurm job by run_benchmark.sh, so the matrix/fidelity jobs that depend
# on it never race to build them.
#   $GLM5_VENV        vLLM+fork (IVF arm): this repo, editable.
#   $GLM5_STOCK_VENV  stock vLLM (kpool arm): the unmodified PyPI release,
#                     nothing from this repo.
set -euo pipefail
REPO="${GLM5_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)}"
VENV="${GLM5_VENV:-${REPO}/.venv}"
STOCK_VENV="${GLM5_STOCK_VENV:-${REPO}/.venv-stock}"
STOCK_VLLM_VERSION="${STOCK_VLLM_VERSION:-0.32.0}"
export PATH="${HOME}/.local/bin:${PATH}"
# `auto` detects the CUDA wheel from the GPU node's driver. Override (e.g.
# TORCH_BACKEND=cu130) if auto picks a torch that ABI-mismatches the
# precompiled vLLM wheel -- `import vllm` would fail here, blocking the jobs.
TORCH_BACKEND="${TORCH_BACKEND:-auto}"

if "${VENV}/bin/python" -c "import vllm" >/dev/null 2>&1; then
  echo "[build] fork venv ready: vllm $("${VENV}/bin/python" -c 'import vllm; print(vllm.__version__)')"
else
  echo "[build] building ${VENV} (VLLM_USE_PRECOMPILED; IVF is pure Python/Triton)"
  uv venv --python 3.12 "${VENV}"
  ( cd "${REPO}" && VLLM_USE_PRECOMPILED=1 \
      VLLM_PRECOMPILED_WHEEL_LOCATION="${VLLM_PRECOMPILED_WHEEL_LOCATION:-https://wheels.vllm.ai/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm-0.32.0-cp38-abi3-manylinux_2_28_x86_64.whl}" \
      uv pip install --python "${VENV}/bin/python" -e . \
        --torch-backend="${TORCH_BACKEND}" )
  "${VENV}/bin/python" -c "import vllm; print('[build] fork vllm', vllm.__version__)"
fi

if "${STOCK_VENV}/bin/python" -c "import vllm" >/dev/null 2>&1; then
  echo "[build] stock venv ready: vllm $("${STOCK_VENV}/bin/python" -c 'import vllm; print(vllm.__version__)')"
else
  echo "[build] building ${STOCK_VENV} (stock vllm==${STOCK_VLLM_VERSION} from PyPI)"
  uv venv --python 3.12 "${STOCK_VENV}"
  # Install from outside the repo so nothing of the fork is picked up.
  ( cd / && uv pip install --python "${STOCK_VENV}/bin/python" \
      "vllm==${STOCK_VLLM_VERSION}" --torch-backend="${TORCH_BACKEND}" )
fi
# The stock arm must not see the fork: vllm resolves to site-packages. Run from
# / so a cwd inside the repo cannot shadow it.
( cd / && "${STOCK_VENV}/bin/python" - "${STOCK_VLLM_VERSION}" "${REPO}" <<'PY'
import os, sys
import vllm
want, repo = sys.argv[1], os.path.realpath(sys.argv[2])
path = os.path.realpath(vllm.__file__)
assert vllm.__version__ == want, f"stock vllm {vllm.__version__} != {want}"
assert not path.startswith(repo + os.sep), f"stock vllm imports the fork: {path}"
print(f"[build] stock vllm {vllm.__version__} at {path}")
PY
)
