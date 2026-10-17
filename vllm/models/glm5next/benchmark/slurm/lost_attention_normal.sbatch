#!/usr/bin/env bash
# IVF lost-attention fidelity sweep — NORMAL (non-preempt) mode.
#
# The #SBATCH partition/qos/gres lines below are BABEL defaults (flame-earlybirds
# / earlybird_qos: 8x H100, TP8). On another cluster edit them, or override at
# submit time, e.g.:
#   sbatch --partition=<p> --qos=<q> --gres=gpu:<type>:8 lost_attention_normal.sbatch
# TP8 needs all 8 cards (4x80<306 GB) and must divide the model's head counts.
#
# Submit from this slurm/ directory, or set GLM5_REPO to the vllm repo root:
#   RULER_SRC=/path/to/ruler sbatch lost_attention_normal.sbatch
# RULER_SRC is required; HF_TOKEN only if the model is gated.
#SBATCH --job-name=glm5-lostattn-normal
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
BODY="${GLM5_REPO}/vllm/models/glm5next/benchmark/slurm/_run_body.sh"
[ -f "${BODY}" ] || { echo "no body at ${BODY}; set GLM5_REPO to the repo root" >&2; exit 2; }
exec bash "${BODY}"
