#!/usr/bin/env bash
# Sourced by batches; submit from the rrl_bayesian repository root.
set -euo pipefail
RRL_ROOT="${RRL_ROOT:-${SLURM_SUBMIT_DIR:?Submit using sbatch from rrl_bayesian}}"
cd "$RRL_ROOT"
[[ -f experiments/compare_pong.py ]] || { echo 'Not the rrl_bayesian root' >&2; exit 2; }
command -v conda >/dev/null || { echo 'Load your cluster Conda module first' >&2; exit 2; }
RRL_CONDA_BASE="$(conda info --base)"
source "$RRL_CONDA_BASE/etc/profile.d/conda.sh"
conda activate "${RRL_CONDA_ENV:-rrl-pong}"
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export SDL_VIDEODRIVER=dummy
export SDL_AUDIODRIVER=dummy
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONPATH="$RRL_ROOT${PYTHONPATH:+:$PYTHONPATH}"
RRL_LOG_DIR="${RRL_LOG_DIR:-$RRL_ROOT/results/cluster_jobs/${SLURM_JOB_ID:-manual}}"
mkdir -p "$RRL_LOG_DIR"
python -m pip check
python -m pip freeze > "$RRL_LOG_DIR/pip-freeze.txt"
conda list --explicit > "$RRL_LOG_DIR/conda-explicit.txt"
hostname
python --version
echo "job=${SLURM_JOB_ID:-manual} root=$RRL_ROOT"
