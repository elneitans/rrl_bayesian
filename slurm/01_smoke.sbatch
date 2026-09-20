#!/usr/bin/env bash
#SBATCH --job-name=rrl-smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=rrl-smoke-%j.out
#SBATCH --partition=ialab-low
#SBATCH --account=default-account
#SBATCH --qos=regular
set -euo pipefail
source "${RRL_ROOT:-${SLURM_SUBMIT_DIR:?}}/slurm/common.sh"
python -m experiments.cluster_launch --kind smoke --stage train \
  --output "${RRL_OUTPUT:?Set RRL_OUTPUT to a new persistent campaign directory}"
