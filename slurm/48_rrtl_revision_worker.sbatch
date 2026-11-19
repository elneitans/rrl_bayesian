#!/usr/bin/env bash
#SBATCH --comment="Experimento rrtl. Supervisor: Guillermo Puebla"
#SBATCH --partition=ialab-low
#SBATCH --account=default-account
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --job-name=rrl-revision
#SBATCH --signal=B:USR1@120
set -euo pipefail
export RRL_LOG_DIR="${RRL_REVISION_CONTROL:?}/environment/${SLURM_JOB_ID:?}_${SLURM_ARRAY_TASK_ID:-single}"
source "${RRL_ROOT:?}/slurm/common.sh"
stage=${1:?frozen, continue or report}
python -m experiments.diagnose_rrtl_revision --stage "$stage" \
    --output "$RRL_REVISION_CONTROL/campaign" --group "${SLURM_ARRAY_TASK_ID:-0}"
