#!/usr/bin/env bash
#SBATCH --comment="Experimento rrtl: auditoria de padres ANOVA. Supervisor: Guillermo Puebla"
#SBATCH --partition=ialab-low
#SBATCH --account=default-account
#SBATCH --qos=regular
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --job-name=rrl-confirm-audit
set -euo pipefail
export RRL_LOG_DIR="${RRL_CONFIRM_CONTROL:?}/environment/${SLURM_JOB_ID:?}"
export PYTHONDONTWRITEBYTECODE=1
source "${RRL_ROOT:?}/slurm/common.sh"
exec python "$RRL_CONFIRM_CONTROL/audit_tools/verify_parents.py" \
  --parents "$RRL_CONFIRM_CONTROL/parents" \
  --config "$RRL_ROOT/configs/breakout_confirmation_parents_20261109.json" \
  --source-root "$RRL_ROOT" \
  --output "$RRL_CONFIRM_CONTROL/parents_verification.json"
