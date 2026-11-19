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
#SBATCH --job-name=rrl-baseline-group
set -euo pipefail
[[ $# == 2 && "$1" =~ ^(train|evaluate)$ && "$2" =~ ^(pong|breakout)$ ]] || exit 2
mode=$1; game=$2
[[ -L "${RRL_ROOT:?}/results" ]] || exit 3
source "$RRL_ROOT/slurm/common.sh"
if [[ "$game" == pong ]]; then
  campaign="$RRL_ROOT/results/pong_ocatari_2m_extra12_20261026_01"; total=12
else
  campaign="$RRL_ROOT/results/breakout_baseline_15_linux_20261026_01"; total=10
fi
first=0; last=$((total-1))
if [[ "$mode" == train ]]; then
  group=${SLURM_ARRAY_TASK_ID:?}
  if [[ "$game" == pong ]]; then
    [[ "$group" =~ ^[01]$ ]] || exit 2
    first=$((5+group*4)); last=$((first+3))
  else
    [[ "$group" =~ ^[012]$ ]] || exit 2
    first=$((group*4)); last=$((first+3))
  fi
  if (( last>=total )); then last=$((total-1)); fi
fi
worker=''; stopping=0; overall=0
stop_worker() { stopping=1; if [[ -n "$worker" ]]; then kill -USR1 "$worker" 2>/dev/null || true; fi; }
trap stop_worker USR1 TERM
for ((cell=first; cell<=last; cell++)); do
  (( stopping==0 )) || exit 3
  remaining=$(( ${SLURM_JOB_END_TIME:?} - $(date +%s) - 180 ))
  (( remaining>180 )) || exit 3
  echo "mode=$mode game=$game cell=$cell"
  if [[ "$mode" == train ]]; then
    seconds=21480; if (( remaining<seconds )); then seconds=$remaining; fi
    python -m experiments.compare_pong_progressive --stage resume --campaign "$campaign" --cell "$cell" --attempt-seconds "$seconds" &
  else
    python -m experiments.compare_pong_progressive --stage evaluate --campaign "$campaign" --cell "$cell" &
  fi
  worker=$!; code=0
  wait "$worker" || code=$?
  if (( stopping )); then wait "$worker" || true; exit 3; fi
  worker=''
  if (( code==3 )); then overall=3
  elif (( code!=0 )); then exit "$code"
  fi
done
if [[ "$mode" == evaluate ]]; then
  code=0
  python -m experiments.compare_pong_progressive --stage report --campaign "$campaign" || code=$?
  if (( code==3 )); then overall=3
  elif (( code!=0 )); then exit "$code"
  fi
fi
exit "$overall"
