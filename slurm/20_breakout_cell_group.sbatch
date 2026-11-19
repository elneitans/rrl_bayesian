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
#SBATCH --job-name=rrl-breakout-group
#SBATCH --output=rrl-breakout-group-%A_%a.out
set -euo pipefail
[[ $# == 2 && "$1" =~ ^(train|evaluate)$ ]] || { echo 'Usage: SCRIPT train|evaluate CAMPAIGN'; exit 2; }
mode="$1"
campaign="$2"
source "${RRL_ROOT:?}/slurm/common.sh"
group="${SLURM_ARRAY_TASK_ID:?}"
[[ "$group" =~ ^[012]$ ]] || exit 2
worker=''
stopping=0
request_stop() {
  stopping=1
  if [[ -n "$worker" ]]; then kill -USR1 "$worker" 2>/dev/null || true; fi
}
trap request_stop USR1 TERM
overall=0
for ((cell=group*4; cell<group*4+4 && cell<10; cell++)); do
  (( stopping == 0 )) || exit 3
  echo "mode=$mode cell=$cell campaign=$campaign"
  if [[ "$mode" == train ]]; then
    attempt=21480
    if [[ -n "${SLURM_JOB_END_TIME:-}" ]]; then
      remaining=$((SLURM_JOB_END_TIME - $(date +%s) - 180))
      (( remaining > 0 )) || exit 3
      if (( remaining < attempt )); then attempt=$remaining; fi
    fi
    python -m experiments.compare_pong_progressive --stage resume \
      --campaign "$campaign" --cell "$cell" --attempt-seconds "$attempt" &
  else
    python -m experiments.compare_pong_progressive --stage evaluate \
      --campaign "$campaign" --cell "$cell" --point 500000 --group final_development &
  fi
  worker=$!
  code=0
  wait "$worker" || code=$?
  if (( stopping )); then
    wait "$worker" || true
    exit 3
  fi
  worker=''
  if (( code == 3 )); then overall=3
  elif (( code != 0 )); then exit "$code"
  fi
done
exit "$overall"
