#!/usr/bin/env bash
# A/B matrix and fidelity sweep side by side on one 8-GPU node, 4 GPUs each.
#
# One allocation instead of two 4-GPU jobs cuts GPU fragmentation. The header
# matches the cluster's preemptible template (~/scripts/train.sh). On requeue
# this script starts both steps again and each resumes from its checkpoints.
# Required env (export before sbatch, --export=ALL): GLM5_REPO, GLM5_VENV,
# SHARED_MODEL_DIR, SHARED_RULER_DIR, AB_OUT_ROOT, FID_OUT_ROOT.
#   sbatch --time=03-00:00 pair_preempt.sbatch
#SBATCH --job-name=glm5-pair
#SBATCH --output=outputs/%x-%j.log
#SBATCH --time=3-00:00:00
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
: "${GLM5_REPO:?}" "${GLM5_VENV:?}" "${AB_OUT_ROOT:?}" "${FID_OUT_ROOT:?}"
SLURM_DIR="${GLM5_REPO}/vllm/models/glm5next/benchmark/slurm"
export GLM5_TP=4
mkdir -p outputs

# Build the venv once here, so the two steps never race to build it.
srun --exact --ntasks=1 --gres=gpu:1 --cpus-per-task=16 --mem=64G \
  bash "${SLURM_DIR}/_build_venv.sh"

# Distinct serve ports: both steps share SLURM_JOB_ID.
BASE_PORT=$(( 20000 + (SLURM_JOB_ID % 9000) * 2 ))
step() {  # name body out_root port
  srun --exact --ntasks=1 --gres=gpu:4 --cpus-per-task=64 --mem=806320M \
    --output="outputs/%x-%j-$1.log" --open-mode=append \
    env OUT_ROOT="$3" GLM5_PORT="$4" bash "${SLURM_DIR}/$2" &
}
step ab _matrix_body.sh "${AB_OUT_ROOT}" "${BASE_PORT}"
step fid _run_body.sh "${FID_OUT_ROOT}" "$(( BASE_PORT + 1 ))"
wait
