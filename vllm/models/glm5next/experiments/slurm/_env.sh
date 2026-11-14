#!/usr/bin/env bash
# Sourced by both step bodies: venv, CUDA toolkit on PATH, caches.
set -euo pipefail
: "${GLM5_REPO:?}" "${GLM5_VENV:?}" "${GLM5_PORT:?}"
EXP_DIR="${GLM5_REPO}/vllm/models/glm5next/experiments"
export PATH="${HOME}/.local/bin:${PATH}"
# fp8 DeepGEMM JIT-compiles with nvcc at the first request; compute nodes ship
# the toolkit off PATH.
if [ -z "${CUDA_HOME:-}" ]; then
  for c in /usr/local/cuda /usr/local/cuda-13.* /usr/local/cuda-13 /usr/local/cuda-12.*; do
    [ -x "${c}/bin/nvcc" ] && { export CUDA_HOME="${c}"; break; }
  done
fi
[ -n "${CUDA_HOME:-}" ] && export PATH="${CUDA_HOME}/bin:${PATH}"
command -v nvcc >/dev/null || echo "[$(date +%T)] WARNING: nvcc not found"
source "${GLM5_VENV}/bin/activate"
echo "[$(date +%T)] host=$(hostname) gpus=${CUDA_VISIBLE_DEVICES:-?} repo=${GLM5_REPO}" \
  "sha=$(git -C "${GLM5_REPO}" rev-parse --short HEAD)"
python -c "import vllm; print('vllm', vllm.__version__)"
