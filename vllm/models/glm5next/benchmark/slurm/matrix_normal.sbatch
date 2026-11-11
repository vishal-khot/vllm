#!/usr/bin/env bash
# A/B benchmark matrix (kpool vs IVF) — NORMAL (non-preempt) mode.
#
# The #SBATCH partition/qos/gres lines below are BABEL defaults. On another
# cluster either edit them, or override at submit time without editing, e.g.:
#   sbatch --partition=<p> --qos=<q> --gres=gpu:<type>:8 matrix_normal.sbatch
# (later --flags win over the #SBATCH defaults).
#
# Submit from this slurm/ directory, or set GLM5_REPO to the vllm repo root:
#   RULER_SRC=/path/to/ruler sbatch matrix_normal.sbatch
#   RULER_SRC=/path/to/ruler GLM5_REPO=/path/to/vllm sbatch matrix_normal.sbatch
# RULER_SRC is required; HF_TOKEN only if the model is gated. GLM5_ONLY=<ids>
# restricts to a subset of runs.
#SBATCH --job-name=glm5-ab-normal
#SBATCH --partition=flame-earlybirds
#SBATCH --qos=earlybird_qos
#SBATCH --gres=gpu:H100:8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=500G
#SBATCH --time=2-00:00:00
#SBATCH --export=ALL
#SBATCH --output=%x-%j.out

set -euo pipefail
: "${GLM5_TP:=8}"; export GLM5_TP
: "${GLM5_REPO:=$(cd "${SLURM_SUBMIT_DIR:-$(pwd)}/../../../../.." && pwd)}"
BODY="${GLM5_REPO}/vllm/models/glm5next/benchmark/slurm/_matrix_body.sh"
[ -f "${BODY}" ] || { echo "no body at ${BODY}; set GLM5_REPO to the repo root" >&2; exit 2; }
exec bash "${BODY}"
