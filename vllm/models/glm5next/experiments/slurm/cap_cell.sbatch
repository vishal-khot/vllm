#!/usr/bin/env bash
# Experiment 1's remaining capacity cells only (capacity_run.py resumes
# $CAP_OUT; $CAP_ONLY lists the cells), with a host-memory log, since two 1M
# IVF serves ended this step without a word. A whole 8-GPU node, so no other
# job shares its host memory; the serve stays TP=4 on GPUs 0-3, as in every
# other capacity cell. Header as layout_cap.sbatch otherwise. Submit with
# submit_cap.sh.
#SBATCH --job-name=glm5-cap
#SBATCH --output=outputs/%x-%j.log
#SBATCH --time=1-00:00:00
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
: "${GLM5_REPO:?}" "${GLM5_VENV:?}" "${CAP_OUT:?}" "${CAP_ONLY:?}"
mkdir -p outputs
srun --exact --ntasks=1 --gres=gpu:8 --cpus-per-task=128 --mem=1612640M \
  env GLM5_PORT="$(( 30000 + (SLURM_JOB_ID % 9000) * 2 ))" \
  bash "${GLM5_REPO}/vllm/models/glm5next/experiments/slurm/_cap_cell_body.sh"
