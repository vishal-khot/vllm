#!/usr/bin/env bash
# Experiment 5 (emeans_bench.py, one process per GPU) on the emeans-v0.32
# worktree. It imports that worktree's vllm ahead of venv_exp's editable one
# (PYTHONPATH), with the compiled artifacts copied by submit_emeans.sh; the
# venv itself is not rebuilt. Submit with submit_emeans.sh.
#SBATCH --job-name=glm5-emeans
#SBATCH --output=outputs/%x-%j.log
#SBATCH --time=1-00:00:00
#SBATCH --partition=lowprio
#SBATCH --requeue
#SBATCH --signal=SIGTERM@128
#SBATCH --account=ci
#SBATCH --qos=ci
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-gpu=201580M
#SBATCH --export=ALL
#SBATCH --open-mode=append

set -euo pipefail
: "${GLM5_REPO:?}" "${GLM5_VENV:?}" "${EMEANS_OUT:?}" "${KEYS_DIR:?}"
mkdir -p outputs
srun --exact --ntasks=1 --gres=gpu:2 --cpus-per-task=32 --mem=403160M \
  env GLM5_PORT="$(( 30000 + (SLURM_JOB_ID % 9000) * 2 ))" \
  bash "${GLM5_REPO}/vllm/models/glm5next/experiments/slurm/_emeans_body.sh"
