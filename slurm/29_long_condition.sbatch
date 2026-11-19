#!/usr/bin/env bash
#SBATCH --comment="Experimento rrtl. Supervisor: Guillermo Puebla"
#SBATCH --partition=ialab-low
#SBATCH --account=default-account
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --time=24:00:00
#SBATCH --signal=B:USR1@120
#SBATCH --job-name=rrl-long-condition
#SBATCH --output=rrl-long-condition-%j.out
set -euo pipefail
[[ $# == 1 ]] || { echo 'Usage: SCRIPT CAMPAIGN'; exit 2; }
campaign=$1
source "${RRL_ROOT:?}/slurm/common.sh"
exec 9>"$campaign/.condition_worker.lock"
flock -n 9 || { echo 'Condition already has an active worker'; exit 2; }
control="${RRL_CONTROL_SCRIPTS:-$RRL_ROOT/slurm}"
cells=$(python - "$campaign" <<'PY'
import sys
from pathlib import Path
from bayesian_rrtl.online.protocol import load
from experiments.pong_analysis_runtime import checked_read
p=Path(sys.argv[1]);m=load(p)
for index,cell in enumerate(m['cells']):
    status=p/cell['id']/'status.json'
    if not status.exists():
        print(index)
    else:
        s=checked_read(status)
        if s['counter']!=2000000 or s['state']!='completed':print(index)
PY
)
worker=''; stopping=0
stop_worker() { stopping=1; if [[ -n "$worker" ]]; then kill -USR1 "$worker" 2>/dev/null || true; fi; }
trap stop_worker USR1 TERM
for cell in $cells; do
  (( stopping == 0 )) || exit 3
  (( ${SLURM_JOB_END_TIME:?} - $(date +%s) > 900 )) || exit 3
  echo "Starting or resuming cell=$cell campaign=$campaign"
  SLURM_ARRAY_TASK_ID="$cell" bash "$control/28_long_cell.sbatch" "$campaign" &
  worker=$!; code=0
  wait "$worker" || code=$?
  if (( stopping )); then wait "$worker" || true; exit 3; fi
  worker=''
  (( code==0 )) || exit "$code"
done
echo 'All declared training cells have completed; evaluation and verified report remain required.'
