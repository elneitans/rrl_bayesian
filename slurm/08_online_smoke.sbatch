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
#SBATCH --job-name=rrl-online-smoke
#SBATCH --output=rrl-online-smoke-%j.out
set -euo pipefail
source "${RRL_ROOT:-${SLURM_SUBMIT_DIR:?}}/slurm/common.sh"
# Explicitly invoked discardable Linux recovery: six small cells, 64 decisions each.
exec python -m experiments.validate_pong_online_recovery --atari "$@"
