#!/usr/bin/env bash
# Experiment 3 (layout_bench.py, one process per GPU), then the capacity
# bisection's remaining cells (capacity_run.py resumes $CAP_OUT, TP=4), on one
# 4-GPU allocation. Header as exp_pair.sbatch with half the GPUs. A requeue
# reruns the step body, which resumes both from their records. Submit with
# submit_layout.sh.
#SBATCH --job-name=glm5-layout
#SBATCH --output=outputs/%x-%j.log
#SBATCH --time=2-00:00:00
#SBATCH --partition=lowprio
#SBATCH --requeue
#SBATCH --signal=SIGTERM@128
#SBATCH --account=ci
#SBATCH --qos=ci
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-gpu=201580M
#SBATCH --export=ALL
#SBATCH --open-mode=append

set -euo pipefail
: "${GLM5_REPO:?}" "${GLM5_VENV:?}" "${CAP_OUT:?}" "${LAYOUT_OUT:?}" "${KEYS_DIR:?}"
EXP_SLURM="${GLM5_REPO}/vllm/models/glm5next/experiments/slurm"
mkdir -p outputs

# Idempotent: the venv from job 3055563 is reused (editable install).
srun --exact --ntasks=1 --gres=gpu:1 --cpus-per-task=16 --mem=64G \
  bash "${GLM5_REPO}/vllm/models/glm5next/benchmark/slurm/_build_venv.sh"

srun --exact --ntasks=1 --gres=gpu:4 --cpus-per-task=64 --mem=806320M \
  env GLM5_PORT="$(( 30000 + (SLURM_JOB_ID % 9000) * 2 ))" \
  bash "${EXP_SLURM}/_layout_cap_body.sh"
