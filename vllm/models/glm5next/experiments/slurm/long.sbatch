#!/usr/bin/env bash
# Experiment 5 (long_run.py): a 2M context, kpool vs IVF, RULER accuracy and
# prefill time. One 8-GPU node, two TP=4 halves of the samples side by side.
# Header as exp_pair.sbatch. A requeue reruns both halves, which skip finished
# requests. Submit with submit_long.sh.
#SBATCH --job-name=glm5-long
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
: "${GLM5_REPO:?}" "${GLM5_VENV:?}" "${LONG_OUT:?}"
EXP_SLURM="${GLM5_REPO}/vllm/models/glm5next/experiments/slurm"
mkdir -p outputs

srun --exact --ntasks=1 --gres=gpu:1 --cpus-per-task=16 --mem=64G \
  bash "${GLM5_REPO}/vllm/models/glm5next/benchmark/slurm/_build_venv.sh"

BASE_PORT=$(( 30000 + (SLURM_JOB_ID % 9000) * 2 ))
for half in 0 1; do
  srun --exact --ntasks=1 --gres=gpu:4 --cpus-per-task=64 --mem=806320M \
    --output="outputs/%x-%j-h${half}.log" --open-mode=append \
    env GLM5_PORT="$(( BASE_PORT + half ))" LONG_HALF="${half}" \
    bash "${EXP_SLURM}/_long_body.sh" &
done
wait
