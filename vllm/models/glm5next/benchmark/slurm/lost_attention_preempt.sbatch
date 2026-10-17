#!/usr/bin/env bash
# IVF lost-attention fidelity sweep — PREEMPT mode.
#
# The #SBATCH partition/qos/gres lines below are BABEL defaults (preempt /
# preempt_qos: 4x H200, TP4 — the config the Modal runs used). On another
# cluster edit them, or override at submit time, e.g.:
#   sbatch --partition=<p> --qos=<q> --gres=gpu:<type>:4 lost_attention_preempt.sbatch
# The job can be preempted; --requeue restarts it (model restages/resumes,
# cells re-measure from scratch).
#
# Submit from this slurm/ directory, or set GLM5_REPO to the vllm repo root:
#   RULER_SRC=/path/to/ruler sbatch lost_attention_preempt.sbatch
# RULER_SRC is required; HF_TOKEN only if the model is gated.
#SBATCH --job-name=glm5-lostattn-preempt
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --gres=gpu:H200:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --export=ALL
#SBATCH --open-mode=append
#SBATCH --output=%x-%j.out

set -euo pipefail
export GLM5_TP=4
: "${GLM5_REPO:=$(cd "${SLURM_SUBMIT_DIR:-$(pwd)}/../../../../.." && pwd)}"
BODY="${GLM5_REPO}/vllm/models/glm5next/benchmark/slurm/_run_body.sh"
[ -f "${BODY}" ] || { echo "no body at ${BODY}; set GLM5_REPO to the repo root" >&2; exit 2; }
exec bash "${BODY}"
