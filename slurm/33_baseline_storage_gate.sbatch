#!/usr/bin/env bash
#SBATCH --comment="Experimento rrtl. Supervisor: Guillermo Puebla"
#SBATCH --partition=ialab-low
#SBATCH --account=default-account
#SBATCH --qos=regular
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --job-name=rrl-baseline-authorize
set -euo pipefail
[[ -L "${RRL_ROOT:?}/results" ]] || { echo 'Verified archive migration is not complete; do not start baseline'; exit 3; }
source "$RRL_ROOT/slurm/common.sh"
python -m experiments.compare_pong_progressive --stage authorize \
  --campaign "$RRL_ROOT/results/pong_ocatari_2m_extra12_20261026_01" --stop-at 2000000 \
  --reason 'Continue user-requested 2M experiment after verified archive migration.'
python -m experiments.compare_pong_progressive --stage authorize \
  --campaign "$RRL_ROOT/results/breakout_baseline_15_linux_20261026_01" --game Breakout --stop-at 2000000 \
  --reason 'Continue the first ten baseline seeds to 2M after verified archive migration.'
