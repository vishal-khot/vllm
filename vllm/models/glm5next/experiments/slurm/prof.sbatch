#!/usr/bin/env bash
# Experiment 4 (prefill_profile.py, TP=4): one profiled served prefill per
# cell. Header as layout_cap.sbatch. A requeue reruns the body, which skips
# finished cells. Submit with submit_prof.sh.
#SBATCH --job-name=glm5-prof
#SBATCH --output=outputs/%x-%j.log
#SBATCH --time=1-00:00:00
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
: "${GLM5_REPO:?}" "${GLM5_VENV:?}" "${PROF_OUT:?}"
EXP_SLURM="${GLM5_REPO}/vllm/models/glm5next/experiments/slurm"
mkdir -p outputs

srun --exact --ntasks=1 --gres=gpu:1 --cpus-per-task=16 --mem=64G \
  bash "${GLM5_REPO}/vllm/models/glm5next/benchmark/slurm/_build_venv.sh"

srun --exact --ntasks=1 --gres=gpu:4 --cpus-per-task=64 --mem=806320M \
  env GLM5_PORT="$(( 30000 + (SLURM_JOB_ID % 9000) * 2 ))" \
  bash "${EXP_SLURM}/_prof_body.sh"
