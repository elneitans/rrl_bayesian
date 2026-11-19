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
#SBATCH --signal=B:USR1@180
#SBATCH --job-name=rrl-break-1m-calendar
set -euo pipefail
[[ $# == 1 && "$1" =~ ^(corrected_common|cart|extra_trees|bart)$ ]] || exit 2
method=$1
export RRL_LOG_DIR="${RRL_CONTROL:?}/environment_1m/${SLURM_JOB_ID:?}"
source "${RRL_ROOT:?}/slurm/common.sh"
campaign="${RRL_BREAK_1M_ROOT:?}/$method"
exec 8>"$campaign/.calendar_worker.lock"
flock -n 8 || exit 3
worker=''; stopping=0; overall=0
stop_worker() { stopping=1; if [[ -n "$worker" ]]; then kill -USR1 "$worker" 2>/dev/null || true; fi; }
trap stop_worker USR1 TERM
for ((cell=0;cell<10;cell++)); do
  (( stopping==0 )) || exit 3
  (( ${SLURM_JOB_END_TIME:?} - $(date +%s) > 900 )) || exit 3
  # Same group lock as training excludes any subsequent recovery of its cells.
  exec 9>"$campaign/.group-$((cell/2)).lock"
  flock -n 9 || exit 3
  counter=$(python - "$campaign" "$cell" <<'PY'
import sys
from pathlib import Path
from bayesian_rrtl.online.protocol import load, select_cell
from experiments.pong_analysis_runtime import checked_read
p=Path(sys.argv[1]);c=select_cell(load(p),sys.argv[2]);f=p/c['id']/'status.json'
print(checked_read(f)['counter'] if f.exists() else 0)
PY
)
  # Only close completed 1M models. A failed/partial training group is recovered
  # separately; do not evaluate mutable frontiers or silently train beyond 1M.
  if (( counter!=1000000 )); then overall=3; exec 9>&-; continue; fi
  python -m experiments.compare_pong_progressive --stage evaluate --campaign "$campaign" --cell "$cell" &
  worker=$!; rc=0
  wait "$worker" || rc=$?
  if (( stopping )); then wait "$worker" || true; exit 3; fi
  worker=''
  exec 9>&-
  if (( rc==3 )); then overall=3
  elif (( rc!=0 )); then exit "$rc"
  fi
done
export RRL_PROGRESS_REPORT="$campaign/calendar_progress_1m.json"
rc=0
bash "$RRL_CONTROL/slurm/39_breakout_1m_report.sbatch" "$method" || rc=$?
(( rc==0 || rc==3 )) || exit "$rc"
(( overall==0 && rc==0 )) || exit 3
