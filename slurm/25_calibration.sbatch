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
#SBATCH --job-name=rrl-calibration
#SBATCH --output=rrl-calibration-%A_%a.out
set -euo pipefail
source "${RRL_ROOT:?}/slurm/common.sh"
games=(pong breakout)
methods=(cart extra_trees bart)
index=${SLURM_ARRAY_TASK_ID:?}
[[ "$index" =~ ^[0-5]$ ]] || exit 2
campaign="${RRL_CALIBRATION_RESULTS:-$RRL_ROOT/results}/${games[index/3]}_${methods[index%3]}_cost_calibration_20261027"
python -m experiments.compare_pong_progressive --stage resume \
  --campaign "$campaign" --cell 0 --attempt-seconds 71400
python -m experiments.compare_pong_progressive --stage evaluate --campaign "$campaign" --cell 0
python -m experiments.compare_pong_progressive --stage report --campaign "$campaign"
python -m experiments.compare_pong_progressive --stage measurements --campaign "$campaign"
