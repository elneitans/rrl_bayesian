#!/usr/bin/env bash
#SBATCH --comment="Experimento RRTL BART Breakout. Supervisor: Guillermo Puebla"
#SBATCH --partition=ialab-low
#SBATCH --account=default-account
#SBATCH --qos=regular
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --signal=B:USR1@180
#SBATCH --job-name=rrl-bart-ablation
set -euo pipefail

export PYTHONDONTWRITEBYTECODE=1
export RRL_LOG_DIR="${RRL_BART_CONTROL:?}/environment/${SLURM_JOB_ID:?}"
source "${RRL_ROOT:?}/slurm/common.sh"

config="${RRL_BART_CONFIG:-$RRL_ROOT/configs/breakout_bart_ablation_20261110.json}"
entry="from ${RRL_BART_RUNNER:-experiments.breakout_bart_ablation} import main; main()"
case "${1:?stage required}" in
  stability_prepare)
    python -m pytest -q tests/test_bart_stability.py tests/test_breakout_bart_ablation.py \
      tests/test_bart_ensemble.py tests/test_general_bart.py tests/test_pong_online.py
    python -c "$entry" smoke --output "$RRL_BART_CONTROL/smoke" --config "$config" --historical "$RRL_BART_HISTORY"
    exec python -c "$entry" prepare --output "$RRL_BART_CONTROL/pilot" --config "$config" --historical "$RRL_BART_HISTORY"
    ;;
  stability_paired)
    exec python -c "$entry" run --output "$RRL_BART_CONTROL/pilot" --config "$config" \
      --task "${SLURM_ARRAY_TASK_ID:?}" --attempt-seconds 85000
    ;;
  preflight)
    python -m pytest -q tests/test_bart_ensemble.py tests/test_breakout_bart_ablation.py
    python -c "$entry" smoke --output "$RRL_BART_CONTROL/smoke" --config "$config" --historical "$RRL_BART_HISTORY"
    python -c "$entry" prepare --output "$RRL_BART_CONTROL/pilot" --config "$config" --historical "$RRL_BART_HISTORY"
    exec python -c "$entry" fixed --output "$RRL_BART_CONTROL" --config "$config" --historical "$RRL_BART_HISTORY"
    ;;
  prepare)
    python -m pytest -q tests/test_breakout_bart_ablation.py
    python -c "$entry" smoke --output "$RRL_BART_CONTROL/smoke" --config "$config" --historical "$RRL_BART_HISTORY"
    exec python -c "$entry" prepare --output "$RRL_BART_CONTROL/pilot" --config "$config" --historical "$RRL_BART_HISTORY"
    ;;
  paired)
    exec python -c "$entry" run --output "$RRL_BART_CONTROL/pilot" --config "$config" \
      --task "${SLURM_ARRAY_TASK_ID:?}" --attempt-seconds 33000
    ;;
  pilot)
    variants=(control warm_start interval_1024 warmup_4096)
    task="${SLURM_ARRAY_TASK_ID:?}"
    exec python -c "$entry" run --output "$RRL_BART_CONTROL/pilot" --config "$config" \
      --variant "${variants[$((task / 3))]}" --index "$((task % 3))"
    ;;
  report)
    exec python -c "$entry" report --output "$RRL_BART_CONTROL/pilot" --config "$config"
    ;;
  *)
    echo "Unknown stage: $1" >&2
    exit 2
    ;;
esac
