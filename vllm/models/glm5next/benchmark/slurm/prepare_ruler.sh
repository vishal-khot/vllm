#!/usr/bin/env bash
# Generate RULER data for the GLM-5.3-Flash benchmark, using the official
# NVIDIA/RULER synthetic pipeline and the GLM tokenizer (so token lengths are
# correct for this model). Output layout matches what the runners read:
#   <OUT>/<task>/validation_<len>.jsonl
#
#   MODEL_DIR=./model OUT=./ruler NUM_SAMPLES=100 bash prepare_ruler.sh
#
# Network: fetches the RULER source tarball (no git) and its source corpora
# (Paul Graham essays, SQuAD, HotpotQA). Idempotent: skips (task,length) pairs
# already generated. Run on a node with internet (a login node) -- NOT inside a
# compute job that may lack egress.
#
# NOTE: GLM-5.3-Flash ships a custom tokenizer; RULER loads it with
# AutoTokenizer. If that needs trust_remote_code and RULER's loader rejects it,
# set RULER_DIR to pre-generated data instead and skip this script.
set -euo pipefail

MODEL_DIR="${MODEL_DIR:?set MODEL_DIR to the GLM model/tokenizer dir}"
OUT="${OUT:-./ruler}"
NUM_SAMPLES="${NUM_SAMPLES:-100}"
RULER_REF="${RULER_REF:-main}"
WORK="${RULER_WORK:-./_ruler_src}"
TASKS="${RULER_TASKS:-niah_single_1 niah_single_2 niah_single_3 niah_multikey_1 niah_multikey_2 niah_multikey_3 niah_multivalue niah_multiquery vt cwe fwe qa_1 qa_2}"
LENGTHS="${RULER_LENGTHS:-4096 8192 16384 32768 65536 131072}"

export PATH="${HOME}/.local/bin:${PATH}"
mkdir -p "${OUT}"
OUT="$(cd "${OUT}" && pwd)"
MODEL_DIR="$(cd "${MODEL_DIR}" && pwd)"

# RULER's prepare.py deps, in an ephemeral uv env (no global installs).
UVW=(--with transformers --with nltk --with wonderwords --with html2text
     --with tenacity --with numpy --with sentencepiece --with huggingface_hub)

# ---- 1. fetch RULER source (tarball; no git clone) ----
SRC="${WORK}/RULER-${RULER_REF}"
if [ ! -d "${SRC}/scripts/data" ]; then
  mkdir -p "${WORK}"
  echo "[ruler] downloading RULER (${RULER_REF}) ..."
  curl -fsSL "https://codeload.github.com/NVIDIA/RULER/tar.gz/refs/heads/${RULER_REF}" \
    | tar -xz -C "${WORK}"
fi
DATA="${SRC}/scripts/data"
[ -d "${DATA}" ] || { echo "[ruler] ERROR: unexpected RULER layout at ${DATA}" >&2; exit 2; }

# ---- 2. source corpora (essays for NIAH/VT/aggregation; QA sets for qa_*) ----
cd "${DATA}"
uv run "${UVW[@]}" python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')" || true
if ls synthetic/json/*.json >/dev/null 2>&1; then
  echo "[ruler] corpora already present"
else
  uv run "${UVW[@]}" python synthetic/json/download_paulgraham_essay.py
  bash synthetic/json/download_qa_dataset.sh
fi

# ---- 3. generate each (task, length) ----
for L in ${LENGTHS}; do
  for T in ${TASKS}; do
    dst="${OUT}/${T}/validation_${L}.jsonl"
    if [ -s "${dst}" ]; then
      echo "[ruler] have ${T}@${L}"
      continue
    fi
    echo "[ruler] generating ${T}@${L} (n=${NUM_SAMPLES}) ..."
    sd="$(mktemp -d)"
    uv run "${UVW[@]}" python prepare.py \
      --save_dir "${sd}" --benchmark synthetic --task "${T}" \
      --tokenizer_path "${MODEL_DIR}" --tokenizer_type hf \
      --max_seq_length "${L}" --model_template_type base \
      --num_samples "${NUM_SAMPLES}" --random_seed 42
    mkdir -p "${OUT}/${T}"
    # RULER writes <save_dir>/<task>/validation.jsonl; find it defensively.
    gen="$(find "${sd}" -name 'validation.jsonl' | head -1)"
    [ -n "${gen}" ] || { echo "[ruler] ERROR: no validation.jsonl for ${T}@${L}" >&2; exit 2; }
    cp "${gen}" "${dst}"
    rm -rf "${sd}"
  done
done
echo "[ruler] done -> ${OUT}"
