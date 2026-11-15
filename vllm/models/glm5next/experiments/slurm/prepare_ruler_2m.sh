#!/usr/bin/env bash
# Experiment 5's RULER data: every task at 2088960 tokens, 20 samples, through
# the benchmark's prepare_ruler.sh unchanged. Run on the login node (it fetches
# packages). nltk >= 3.10 stops each regex match after 5 s, which punkt's
# sentence splitter exceeds on a 2M haystack; a sitecustomize lifts that limit.
#   BASE=/mnt/weka/shrd/ci/haok/glm5 bash prepare_ruler_2m.sh
set -euo pipefail
BASE="${BASE:-/mnt/weka/shrd/ci/haok/glm5}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
OUT="${BASE}/ruler_2m"
FIX="${OUT}/_pyfix"
mkdir -p "${FIX}"
cat > "${FIX}/sitecustomize.py" <<'PY'
# nltk >= 3.10 caps each regex match at 5 s; punkt's sentence splitter takes
# longer than that on a 2M-token haystack. The text is RULER's own corpus.
try:
    import nltk.redos

    nltk.redos.DEFAULT_TIMEOUT = None
except Exception:
    pass
PY
PYTHONPATH="${FIX}" MODEL_DIR="${BASE}/model" OUT="${OUT}" NUM_SAMPLES=20 \
  RULER_LENGTHS=2088960 RULER_WORK="${RULER_WORK:-/mnt/weka/shrd/ci/haok/glm5-ivf/_ruler_src}" \
  UV_CACHE_DIR="${BASE}/uvcache_exp" \
  bash "${REPO}/vllm/models/glm5next/benchmark/slurm/prepare_ruler.sh"
