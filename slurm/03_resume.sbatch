#!/usr/bin/env bash
#SBATCH --job-name=rrl-resume
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --time=12:00:00
#SBATCH --output=rrl-resume-%j.out
#SBATCH --partition=ialab-low
#SBATCH --account=default-account
#SBATCH --qos=regular
set -euo pipefail
source "${RRL_ROOT:-${SLURM_SUBMIT_DIR:?}}/slurm/common.sh"
RRL_EXTRA=()
if [[ -n "${RRL_TRAINING_SECONDS:-}" ]]; then
  RRL_EXTRA+=(--training-budget-seconds "$RRL_TRAINING_SECONDS")
fi
if [[ -n "${RRL_EVALUATION_SECONDS:-}" ]]; then
  RRL_EXTRA+=(--evaluation-budget-seconds "$RRL_EVALUATION_SECONDS")
fi
if [[ ${#RRL_EXTRA[@]} -gt 0 ]]; then
  RRL_EXTRA+=(--budget-reason "${RRL_BUDGET_REASON:?Explain the authorized increase}")
fi
python -m experiments.cluster_launch --stage resume \
  --output "${RRL_OUTPUT:?Set RRL_OUTPUT to the existing campaign}" "${RRL_EXTRA[@]}"
