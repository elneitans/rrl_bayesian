#!/usr/bin/env bash
#SBATCH --comment="Capstone Breakout BART cobertura del replay y control pareado. Supervisor: Guillermo Puebla"
#SBATCH --partition=ialab-low
#SBATCH --account=default-account
#SBATCH --qos=regular
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --signal=B:USR1@180
#SBATCH --job-name=bart-replay
set -euo pipefail

export PYTHONDONTWRITEBYTECODE=1
export RRL_LOG_DIR="${RRL_BART_COVERAGE:?}/environment/${SLURM_JOB_ID:?}"
source "${RRL_ROOT:?}/slurm/common.sh"
stage=${1:?stage required}
if [[ "$stage" == prepare ]]; then
    python -m pytest -q tests/test_bart_stability.py tests/test_bart_replay_coverage.py
fi
exec python -m experiments.breakout_bart_replay_coverage "$stage" \
    --output "$RRL_BART_COVERAGE/results" --config "$RRL_ROOT/configs/breakout_bart_replay_coverage_20261118.json" \
    --historical "$RRL_BART_CADENCE/pilot/interval_1024" \
    --stability "$RRL_BART_STABILITY" --stability-config "$RRL_ROOT/configs/breakout_bart_stability_1m_20261114.json" \
    --index "${SLURM_ARRAY_TASK_ID:-0}"
