#!/usr/bin/env bash
# A/B benchmark matrix (kpool vs IVF) — PREEMPT mode.
#
# #SBATCH lines follow the cluster's preemptible job template: a low-priority
# partition with --requeue and --signal, so a preempted job is requeued and the
# runner resumes (only runs without a verdict.json are re-measured). run_benchmark.sh
# overrides --partition/--qos/--gres at submit and the account comes from
# SBATCH_ACCOUNT, so those are safe defaults; tune --cpus-per-gpu/--mem-per-gpu to
# the node. Direct use: (ensure an outputs/ dir exists)
#   SBATCH_ACCOUNT=ci GLM5_REPO=/path/to/vllm SHARED_MODEL_DIR=... SHARED_RULER_DIR=... \
#     sbatch --partition=lowprio --qos=ci --gres=gpu:8 matrix_preempt.sbatch
#SBATCH --job-name=glm5-ab
#SBATCH --output=outputs/%x-%j.log
#SBATCH --time=2-00:00:00
#SBATCH --partition=lowprio
#SBATCH --requeue
#SBATCH --signal=SIGTERM@128
#SBATCH --account=ci
#SBATCH --qos=ci
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-gpu=201580M
#SBATCH --export=ALL
#SBATCH --open-mode=append

set -euo pipefail
: "${GLM5_TP:=8}"; export GLM5_TP
: "${GLM5_REPO:=$(cd "${SLURM_SUBMIT_DIR:-$(pwd)}/../../../../.." && pwd)}"
BODY="${GLM5_REPO}/vllm/models/glm5next/benchmark/slurm/_matrix_body.sh"
[ -f "${BODY}" ] || { echo "no body at ${BODY}; set GLM5_REPO to the repo root" >&2; exit 2; }
exec bash "${BODY}"
