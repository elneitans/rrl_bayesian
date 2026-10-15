#!/usr/bin/env bash
#SBATCH --comment="Experimento rrtl. Supervisor: Guillermo Puebla"
#SBATCH --job-name=rrl-atari-preflight
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --time=01:00:00
#SBATCH --partition=ialab-low
#SBATCH --account=default-account
#SBATCH --qos=regular
#SBATCH --output=rrl-atari-preflight-%j.out
set -euo pipefail
source "${RRL_ROOT:-${SLURM_SUBMIT_DIR:?}}/slurm/common.sh"
[[ "$(uname -s)" == Linux ]] || { echo 'Pinned Linux validation required' >&2; exit 2; }
export RRL_C_REAL_TEST=1
python -m pytest tests/test_atari_block_c.py tests/test_atari_block_c_real.py -q
python -m experiments.compare_pong_progressive --stage dry-run --protocol configs/atari_comparison_smoke.json > "$RRL_LOG_DIR/atari-smoke-resolved.json"
