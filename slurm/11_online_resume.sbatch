#!/usr/bin/env bash
#SBATCH --comment="Experimento rrtl. Supervisor: Guillermo Puebla"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --time=01:00:00
#SBATCH --partition=ialab-low
#SBATCH --account=default-account
#SBATCH --qos=regular
#SBATCH --signal=B:USR1@120
#SBATCH --job-name=rrl-online-resume
#SBATCH --output=rrl-online-resume-%j.out
set -euo pipefail
source "${RRL_ROOT:-${SLURM_SUBMIT_DIR:?}}/slurm/common.sh"
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  exec python -m experiments.compare_pong_progressive --stage resume --attempt-seconds 3480 --cell "$SLURM_ARRAY_TASK_ID" "$@"
fi
exec python -m experiments.compare_pong_progressive --stage resume --attempt-seconds 3480 "$@"
