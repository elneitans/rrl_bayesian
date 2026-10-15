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
#SBATCH --job-name=rrl-online-preflight
#SBATCH --output=rrl-online-preflight-%j.out
set -euo pipefail
source "${RRL_ROOT:-${SLURM_SUBMIT_DIR:?}}/slurm/common.sh"
[[ "$(uname -s)" == Linux ]] || { echo 'Linux runtime required' >&2; exit 2; }
python -m pytest tests/test_pong_online.py tests/test_cluster_launch.py -q
python -c 'from bayesian_rrtl.ocatari_backend import verified_versions; print(verified_versions())'
python -m experiments.compare_pong_progressive --stage dry-run --protocol configs/pong_online_smoke.json
