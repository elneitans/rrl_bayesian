#!/usr/bin/env bash
#SBATCH --job-name=rrl-preflight
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=rrl-preflight-%j.out
#SBATCH --partition=ialab-low
#SBATCH --account=default-account
#SBATCH --qos=regular
set -euo pipefail
source "${RRL_ROOT:-${SLURM_SUBMIT_DIR:?}}/slurm/common.sh"
[[ "$(uname -s)" == Linux ]] || { echo 'Cluster preflight requires Linux' >&2; exit 2; }
python -m pytest tests/test_cluster_launch.py tests/test_baseline_checkpoint.py \
  tests/test_comparison_protocol.py tests/test_comparison_training.py \
  tests/test_pong_comparison_inputs.py tests/test_pong_comparison_pipeline.py \
  tests/test_ocatari_resume.py -q --junitxml="$RRL_LOG_DIR/preflight.xml"
# Resolve/validate all pilot cells without training or consuming test seeds.
python -m experiments.compare_pong --protocol configs/comparison_pong_pilot.json --dry-run \
  > "$RRL_LOG_DIR/pilot-resolved.json"
