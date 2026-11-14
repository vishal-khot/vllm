#!/usr/bin/env bash
# Both follow-up experiments on one 8-GPU node, 4 GPUs each:
#   cap      capacity_run.py (TP=4 serves)
#   reorder  capture_run.py (TP=4 serve), then reorder_bench.py on each GPU
# Same header as ../../benchmark/slurm/pair_preempt.sbatch. On requeue both
# steps start again and resume from their checkpoints. Submit with submit.sh,
# which sets the env and refuses to reuse existing output dirs.
#SBATCH --job-name=glm5-exp
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
: "${GLM5_REPO:?}" "${GLM5_VENV:?}" "${CAP_OUT:?}" "${REORDER_OUT:?}"
EXP_SLURM="${GLM5_REPO}/vllm/models/glm5next/experiments/slurm"
mkdir -p outputs

# The benchmark's venv builder, pointed at this job's own GLM5_VENV.
srun --exact --ntasks=1 --gres=gpu:1 --cpus-per-task=16 --mem=64G \
  bash "${GLM5_REPO}/vllm/models/glm5next/benchmark/slurm/_build_venv.sh"

# Ports apart from the matrix job's 20000+ range.
BASE_PORT=$(( 30000 + (SLURM_JOB_ID % 9000) * 2 ))
step() {  # name body port
  srun --exact --ntasks=1 --gres=gpu:4 --cpus-per-task=64 --mem=806320M \
    --output="outputs/%x-%j-$1.log" --open-mode=append \
    env GLM5_PORT="$3" bash "${EXP_SLURM}/$2" &
}
step cap _cap_body.sh "${BASE_PORT}"
step reorder _reorder_body.sh "$(( BASE_PORT + 1 ))"
wait
