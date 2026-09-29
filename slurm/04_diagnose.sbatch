#!/usr/bin/env bash
#SBATCH --job-name=rrl-diagnose
#SBATCH --comment="Experimento rrtl. Supervisor: Guillermo Puebla"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --time=03:00:00
#SBATCH --output=rrl-diagnose-%j.out
#SBATCH --partition=ialab-low
#SBATCH --account=default-account
#SBATCH --qos=regular
set -euo pipefail
source "${RRL_ROOT:-${SLURM_SUBMIT_DIR:?}}/slurm/common.sh"
RRL_DIAGNOSTIC_ARGS=()
if [[ -n "${RRL_DIAGNOSTIC_CELL:-}" ]]; then
  RRL_DIAGNOSTIC_ARGS+=(--cell "$RRL_DIAGNOSTIC_CELL")
fi
python -m experiments.diagnose_pong_policy \
  --campaign "${RRL_OUTPUT:?Existing completed or partial campaign}" \
  --output "${RRL_DIAGNOSTIC_OUTPUT:?New diagnostic directory}" \
  --horizon 2000 --seconds-per-cell 600 --gif "${RRL_DIAGNOSTIC_ARGS[@]}"
