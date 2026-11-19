#!/usr/bin/env bash
#SBATCH --comment="Experimento rrtl: controles TD y revisiones protegidas Breakout. Supervisor: Guillermo Puebla"
#SBATCH --partition=ialab-low
#SBATCH --account=default-account
#SBATCH --qos=regular
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --signal=B:USR1@120
#SBATCH --job-name=rrl-improve
set -euo pipefail
export RRL_LOG_DIR="${RRL_IMPROVEMENT_CONTROL:?}/environment/${SLURM_JOB_ID:?}"
source "${RRL_ROOT:?}/slurm/common.sh"
stage="${1:-continue}"
exec python -m experiments.diagnose_rrtl_improvements --stage "$stage" \
    --output "$RRL_IMPROVEMENT_CONTROL/campaign" --group "${SLURM_ARRAY_TASK_ID:-0}"
