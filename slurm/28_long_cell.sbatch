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
#SBATCH --job-name=rrl-long-cell
#SBATCH --output=rrl-long-cell-%A_%a.out
set -euo pipefail
[[ $# == 1 ]] || { echo 'Usage: sbatch --array=... SCRIPT CAMPAIGN'; exit 2; }
campaign=$1
source "${RRL_ROOT:?}/slurm/common.sh"
cell=${SLURM_ARRAY_TASK_ID:?}
worker=''; stopping=0
stop_worker() { stopping=1; if [[ -n "$worker" ]]; then kill -USR1 "$worker" 2>/dev/null || true; fi; }
trap stop_worker USR1 TERM
for ((attempt=0; attempt<40; attempt++)); do
  (( stopping == 0 )) || exit 3
  remaining=$(( ${SLURM_JOB_END_TIME:?} - $(date +%s) - 600 ))
  (( remaining > 180 )) || exit 3
  seconds=1740
  if (( remaining < seconds )); then seconds=$remaining; fi
  python -m experiments.compare_pong_progressive --stage resume \
    --campaign "$campaign" --cell "$cell" --attempt-seconds "$seconds" &
  worker=$!; code=0
  wait "$worker" || code=$?
  if (( stopping )); then wait "$worker" || true; exit 3; fi
  worker=''
  (( code==0 || code==3 )) || exit "$code"
  # Supported retention preserves current/previous/calendar/evaluated checkpoints,
  # metadata chains and all committed transition logs.
  python -m experiments.compare_pong_progressive --stage cleanup --campaign "$campaign" --cell "$cell"
  (( code != 0 )) || exit 0
done
exit 3
