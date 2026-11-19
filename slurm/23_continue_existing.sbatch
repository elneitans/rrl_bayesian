#!/usr/bin/env bash
#SBATCH --comment="Experimento rrtl. Supervisor: Guillermo Puebla"
#SBATCH --partition=ialab-low
#SBATCH --account=default-account
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --time=08:00:00
#SBATCH --signal=B:USR1@120
#SBATCH --job-name=rrl-continue-existing
#SBATCH --output=rrl-continue-%A_%a.out
set -euo pipefail
source "${RRL_ROOT:?}/slurm/common.sh"
case "${RRL_TASK:?}" in
  pong)
    exec python -m experiments.compare_pong_progressive --stage resume \
      --campaign "$RRL_ROOT/results/pong_ocatari_2m_extra12_20261026_01" \
      --cell "${SLURM_ARRAY_TASK_ID:?}" --attempt-seconds 28620 ;;
  breakout)
    exec python -m experiments.compare_pong_progressive --stage resume \
      --campaign "$RRL_ROOT/results/breakout_baseline_15_linux_20261026_01" \
      --cell "${SLURM_ARRAY_TASK_ID:?}" --attempt-seconds 28620 ;;
  breakout-evaluation-repair)
    exec python -m experiments.compare_pong_progressive --stage evaluate \
      --campaign "$RRL_ROOT/results/breakout_cart_10seeds_500k_20261026_01" \
      --cell 8 --point 500000 --group final_development \
      --extend-seconds 1800 --extension-id eval-500k-repair-20261027 \
      --reason 'Complete existing administrative partial evaluations; same checkpoint, policy, horizon and seeds.' ;;
  authorize-breakout)
    exec python -m experiments.compare_pong_progressive --stage authorize \
      --campaign "$RRL_ROOT/results/breakout_baseline_15_linux_20261026_01" \
      --game Breakout --stop-at 2000000 \
      --reason 'User requested all implemented algorithms to 2M; continue first ten existing baseline seeds.' ;;
  *) exit 2 ;;
esac
