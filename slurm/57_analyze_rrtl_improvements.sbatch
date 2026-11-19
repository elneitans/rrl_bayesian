#!/usr/bin/env bash
#SBATCH --comment="Experimento rrtl: analisis de mejoras y controles por semilla. Supervisor: Guillermo Puebla"
#SBATCH --partition=ialab-low
#SBATCH --account=default-account
#SBATCH --qos=regular
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --job-name=rrl-improve-analysis
set -euo pipefail
export RRL_LOG_DIR="${RRL_IMPROVEMENT_CONTROL:?}/environment/${SLURM_JOB_ID:?}"
source "${RRL_ROOT:?}/slurm/common.sh"
exec python -m experiments.analyze_rrtl_improvements --campaign "$RRL_IMPROVEMENT_CONTROL/campaign" \
    --output "$RRL_IMPROVEMENT_CONTROL/analysis"
