#!/usr/bin/env bash
# Build both venvs once, on a GPU node (so --torch-backend=auto picks the CUDA
# wheel). Idempotent: skips a venv whose vllm already imports. Run as its own
# short Slurm job by run_benchmark.sh, so the matrix/fidelity jobs that depend
# on it never race to build them.
#   $GLM5_VENV        vLLM+IVF (ivf arm): the IVF clone ($GLM5_IVF_REPO,
#                     default ../vllm-ivf), editable.
#   $GLM5_STOCK_VENV  stock vLLM (kpool arm): the unmodified PyPI release,
#                     nothing from either clone.
set -euo pipefail
REPO="${GLM5_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)}"
IVF_REPO="${GLM5_IVF_REPO:-$(dirname "${REPO}")/vllm-ivf}"
VENV="${GLM5_VENV:-${IVF_REPO}/.venv}"
STOCK_VENV="${GLM5_STOCK_VENV:-${REPO}/.venv-stock}"
STOCK_VLLM_VERSION="${STOCK_VLLM_VERSION:-0.32.0}"
export PATH="${HOME}/.local/bin:${PATH}"
# `auto` detects the CUDA wheel from the GPU node's driver. Override (e.g.
# TORCH_BACKEND=cu130) if auto picks a torch that ABI-mismatches the
# precompiled vLLM wheel -- `import vllm` would fail here, blocking the jobs.
TORCH_BACKEND="${TORCH_BACKEND:-auto}"

if "${VENV}/bin/python" -c "import vllm" >/dev/null 2>&1; then
  echo "[build] ivf venv ready: vllm $("${VENV}/bin/python" -c 'import vllm; print(vllm.__version__)')"
else
  echo "[build] building ${VENV} (VLLM_USE_PRECOMPILED; IVF is pure Python/Triton)"
  uv venv --python 3.12 "${VENV}"
  ( cd "${IVF_REPO}" && VLLM_USE_PRECOMPILED=1 \
      VLLM_PRECOMPILED_WHEEL_LOCATION="${VLLM_PRECOMPILED_WHEEL_LOCATION:-https://wheels.vllm.ai/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm-0.32.0-cp38-abi3-manylinux_2_28_x86_64.whl}" \
      uv pip install --python "${VENV}/bin/python" -e . \
        --torch-backend="${TORCH_BACKEND}" )
fi

if "${STOCK_VENV}/bin/python" -c "import vllm" >/dev/null 2>&1; then
  echo "[build] stock venv ready: vllm $("${STOCK_VENV}/bin/python" -c 'import vllm; print(vllm.__version__)')"
else
  echo "[build] building ${STOCK_VENV} (stock vllm==${STOCK_VLLM_VERSION} from PyPI)"
  uv venv --python 3.12 "${STOCK_VENV}"
  # Install from outside both clones so nothing of them is picked up.
  ( cd / && uv pip install --python "${STOCK_VENV}/bin/python" \
      "vllm==${STOCK_VLLM_VERSION}" --torch-backend="${TORCH_BACKEND}" )
fi
# The stock arm must see neither clone: vllm resolves to site-packages. Run
# from / so a cwd inside a clone cannot shadow it.
( cd / && "${STOCK_VENV}/bin/python" - "${STOCK_VLLM_VERSION}" "${REPO}" "${IVF_REPO}" <<'PY'
import os, sys
import vllm
want = sys.argv[1]
path = os.path.realpath(vllm.__file__)
assert vllm.__version__ == want, f"stock vllm {vllm.__version__} != {want}"
for repo in map(os.path.realpath, sys.argv[2:]):
    assert not path.startswith(repo + os.sep), f"stock vllm imports {repo}: {path}"
print(f"[build] stock vllm {vllm.__version__} at {path}")
PY
)
# The ivf arm must serve the IVF clone, not the harness tree.
( cd / && "${VENV}/bin/python" - "${IVF_REPO}" <<'PY'
import os, sys
import importlib.util
import vllm
ivf = "vllm.models.glm5next.nvidia.ivf_sparse_indexer"
assert importlib.util.find_spec(ivf), f"ivf vllm has no {ivf}"
repo = os.path.realpath(sys.argv[1])
path = os.path.realpath(vllm.__file__)
assert path.startswith(repo + os.sep), f"ivf vllm is not the IVF clone: {path}"
print(f"[build] ivf vllm {vllm.__version__} at {path}")
PY
)
