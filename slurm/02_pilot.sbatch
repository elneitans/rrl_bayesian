#!/usr/bin/env bash
#SBATCH --job-name=rrl-pilot
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --time=12:00:00
#SBATCH --output=rrl-pilot-%j.out
#SBATCH --partition=ialab-low
#SBATCH --account=default-account
#SBATCH --qos=regular
set -euo pipefail
source "${RRL_ROOT:-${SLURM_SUBMIT_DIR:?}}/slurm/common.sh"
[[ "${RRL_CONFIRM_PILOT:-}" == yes ]] || { echo 'Set RRL_CONFIRM_PILOT=yes after Linux smoke' >&2; exit 2; }
python -m experiments.cluster_launch --kind pilot --stage train \
  --output "${RRL_OUTPUT:?Set RRL_OUTPUT to a new persistent campaign directory}"
