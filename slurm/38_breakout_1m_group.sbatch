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
#SBATCH --job-name=rrl-break-1m-group
set -euo pipefail
[[ $# == 1 && "$1" =~ ^(corrected_common|cart|extra_trees|bart)$ ]] || exit 2
method=$1
export RRL_LOG_DIR="${RRL_CONTROL:?}/environment_1m/${SLURM_JOB_ID:?}_${SLURM_ARRAY_TASK_ID:?}"
source "${RRL_ROOT:?}/slurm/common.sh"
campaign="${RRL_BREAK_1M_ROOT:?}/$method"
group=${SLURM_ARRAY_TASK_ID:?}
[[ "$group" =~ ^[0-4]$ ]] || exit 2
exec 9>"$campaign/.group-$group.lock"
flock -n 9 || { echo 'Duplicate group worker'; exit 2; }
worker=''; stopping=0
stop_worker() { stopping=1; if [[ -n "$worker" ]]; then kill -USR1 "$worker" 2>/dev/null || true; fi; }
trap stop_worker USR1 TERM
run_child() {
  "$@" & worker=$!
  local rc=0
  wait "$worker" || rc=$?
  if (( stopping )); then wait "$worker" || true; worker=''; return 3; fi
  worker=''; return "$rc"
}
for ((cell=group*2;cell<group*2+2;cell++)); do
  while true; do
    (( stopping==0 )) || exit 3
    remaining=$(( ${SLURM_JOB_END_TIME:?} - $(date +%s) - 600 ))
    (( remaining>180 )) || exit 3
    reached=$(python - "$campaign" "$cell" <<'PY'
import sys
from pathlib import Path
from bayesian_rrtl.online.protocol import load, select_cell
from experiments.pong_analysis_runtime import checked_read
p=Path(sys.argv[1]); c=select_cell(load(p),sys.argv[2]); f=p/c['id']/'status.json'
s=checked_read(f) if f.exists() else None
if s and s['counter']<1000000 and s.get('reason')=='training_budget_exhausted':
    raise SystemExit('Cumulative training budget exhausted; review before extending')
print(int(s is not None and s['counter']>=1000000))
PY
)
    [[ "$reached" == 1 ]] && break
    seconds=1740; if (( remaining<seconds )); then seconds=$remaining; fi
    rc=0
    run_child python -m experiments.compare_pong_progressive --stage resume --campaign "$campaign" --cell "$cell" --attempt-seconds "$seconds" || rc=$?
    (( stopping==0 )) || exit 3
    (( rc==0 || rc==3 )) || exit "$rc"
    python -m experiments.compare_pong_progressive --stage cleanup --campaign "$campaign" --cell "$cell"
    # Check the actual counter on every iteration. Reaching an authorized
    # intermediate milestone still has state=partial and CLI exit code 3.
  done
  (( ${SLURM_JOB_END_TIME:?} - $(date +%s) > 900 )) || exit 3
  rc=0
  run_child python -m experiments.compare_pong_progressive --stage evaluate --campaign "$campaign" --cell "$cell" --point 1000000 --group final_development || rc=$?
  (( rc==0 )) || exit "$rc"
done
echo "Group $group complete at 1M with development evaluation: $method"
