#!/usr/bin/env bash
# The listed capacity cells, with host memory and the step's cgroup memory
# events logged every 30 s to $CAP_OUT/hostmem_<job>.log.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
: "${CAP_OUT:?}" "${CAP_ONLY:?}"
# The serve is TP=4, as in every other capacity cell: the first 4 GPUs.
export CUDA_VISIBLE_DEVICES="$(cut -d, -f1-4 <<< "${CUDA_VISIBLE_DEVICES}")"
CG="/sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup | head -1)"
MEMLOG="${CAP_OUT}/hostmem_${SLURM_JOB_ID:-x}.log"
(
  while true; do
    echo "[$(date +%T)] $(free -g | awk '/Mem:/{print "used=" $3 "G avail=" $7 "G"}')" \
      "top=$(ps -eo rss,comm --sort=-rss | sed -n 2,4p | awk '{printf "%s:%dG ", $2, $1/1048576}')" \
      "cg=$(tr '\n' ' ' < "${CG}/memory.events" 2>/dev/null)" \
      "cg_cur=$(( $(cat "${CG}/memory.current" 2>/dev/null || echo 0) / 1073741824 ))G"
    sleep 30
  done
) >> "${MEMLOG}" 2>&1 &
mon=$!
set +e
python -X faulthandler "${EXP_DIR}/capacity_run.py" --out-dir "${CAP_OUT}" \
  --port "${GLM5_PORT}" --only "${CAP_ONLY}"
rc=$?
set -e
kill "${mon}" 2>/dev/null || true
echo "[$(date +%T)] capacity exit ${rc} -> ${CAP_OUT} (memory log ${MEMLOG})"
exit "${rc}"
