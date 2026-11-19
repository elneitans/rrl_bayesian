#!/usr/bin/env bash
#SBATCH --comment="Experimento rrtl. Supervisor: Guillermo Puebla"
#SBATCH --job-name=rrl-evaluate-cells
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --time=12:00:00
#SBATCH --partition=ialab-low
#SBATCH --account=default-account
#SBATCH --qos=regular
#SBATCH --signal=B:USR1@120
#SBATCH --output=rrl-evaluate-cells-%j.out
set -euo pipefail
(( $# >= 2 )) || { echo 'Usage: sbatch slurm/18_evaluate_cells.sbatch CAMPAIGN CELL [CELL ...]' >&2; exit 2; }
source "${RRL_ROOT:-${SLURM_SUBMIT_DIR:?}}/slurm/common.sh"
campaign="$1"
shift
stopping=0
worker=""
request_stop() {
  stopping=1
  if [[ -n "$worker" ]]; then kill -USR1 "$worker" 2>/dev/null || true; fi
}
trap request_stop USR1 TERM
overall=0
for cell in "$@"; do
  (( stopping == 0 )) || exit 3
  echo "Evaluating cell $cell in $campaign"
  python -m experiments.compare_pong_progressive --stage evaluate --campaign "$campaign" --cell "$cell" &
  worker=$!
  code=0
  wait "$worker" || code=$?
  if (( stopping )); then
    wait "$worker" || true
    exit 3
  fi
  worker=""
  if (( code == 3 )); then
    overall=3
  elif (( code != 0 )); then
    exit "$code"
  fi
done
exit "$overall"
