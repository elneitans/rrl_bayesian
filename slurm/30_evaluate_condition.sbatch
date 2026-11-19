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
#SBATCH --job-name=rrl-condition-eval
#SBATCH --output=rrl-condition-eval-%j.out
set -euo pipefail
[[ $# == 1 ]] || exit 2
campaign=$1
source "${RRL_ROOT:?}/slurm/common.sh"
exec 9>"$campaign/.condition_evaluation_worker.lock"
flock -n 9 || exit 2
cells=$(python - "$campaign" <<'PY'
import sys
from pathlib import Path
from bayesian_rrtl.online.protocol import load
from experiments.pong_analysis_runtime import checked_read
p=Path(sys.argv[1]);m=load(p)
for index,cell in enumerate(m['cells']):
    f=p/cell['id']/'status.json'
    if f.exists() and checked_read(f)['counter']>0:print(index)
PY
)
worker=''; stopping=0; overall=0
stop_worker() { stopping=1; if [[ -n "$worker" ]]; then kill -USR1 "$worker" 2>/dev/null || true; fi; }
trap stop_worker USR1 TERM
for cell in $cells; do
  (( stopping == 0 )) || exit 3
  (( ${SLURM_JOB_END_TIME:?} - $(date +%s) > 900 )) || exit 3
  python -m experiments.compare_pong_progressive --stage evaluate --campaign "$campaign" --cell "$cell" &
  worker=$!; code=0
  wait "$worker" || code=$?
  if (( stopping )); then wait "$worker" || true; exit 3; fi
  worker=''
  if (( code==3 )); then overall=3
  elif (( code!=0 )); then exit "$code"
  fi
done
python -m experiments.compare_pong_progressive --stage report --campaign "$campaign"
exit "$overall"
