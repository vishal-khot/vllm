#!/usr/bin/env bash
# Two halves of the A/B matrix side by side on one 8-GPU node, 4 GPUs each.
#
# Same header and resume behaviour as pair_preempt.sbatch, but both steps run
# matrix_run.py: step a over the run ids in ONLY_A, step b over ONLY_B, each
# into its own out dir (summary.json is per out dir). Extra probe env such as
# GLM5_ATTN_PROBE=1 passes through to the serves (--export=ALL).
# Required env: GLM5_REPO, GLM5_VENV, SHARED_MODEL_DIR, SHARED_RULER_DIR,
# ONLY_A, ONLY_B, OUT_A, OUT_B.
#SBATCH --job-name=glm5-split
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
: "${GLM5_REPO:?}" "${GLM5_VENV:?}" "${ONLY_A:?}" "${ONLY_B:?}" "${OUT_A:?}" "${OUT_B:?}"
SLURM_DIR="${GLM5_REPO}/vllm/models/glm5next/benchmark/slurm"
export GLM5_TP=4
mkdir -p outputs

srun --exact --ntasks=1 --gres=gpu:1 --cpus-per-task=16 --mem=64G \
  bash "${SLURM_DIR}/_build_venv.sh"

BASE_PORT=$(( 20000 + (SLURM_JOB_ID % 9000) * 2 ))
step() {  # name only out_root port
  srun --exact --ntasks=1 --gres=gpu:4 --cpus-per-task=64 --mem=806320M \
    --output="outputs/%x-%j-$1.log" --open-mode=append \
    env GLM5_ONLY="$2" OUT_ROOT="$3" GLM5_PORT="$4" \
    bash "${SLURM_DIR}/_matrix_body.sh" &
}
step a "${ONLY_A}" "${OUT_A}" "${BASE_PORT}"
step b "${ONLY_B}" "${OUT_B}" "$(( BASE_PORT + 1 ))"
wait
