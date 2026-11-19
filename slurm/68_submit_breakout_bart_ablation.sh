#!/usr/bin/env bash
set -euo pipefail

export RRL_BART_CONTROL="${1:?Provide the new campaign directory}"
export RRL_ROOT="$RRL_BART_CONTROL/code"
export RRL_BART_HISTORY="${RRL_BART_HISTORY:-$HOME/archive/rrl_bayesian_breakout_1m_20261028/bart}"
export RRL_CONDA_ENV="${RRL_CONDA_ENV:-$HOME/archive/conda_envs/rrl-pong-analysis}"
cd "$RRL_ROOT"
mkdir -p "$RRL_BART_CONTROL/slurm"
ledger="$RRL_BART_CONTROL/jobs.tsv"
exec 9>"$RRL_BART_CONTROL/submission.lock"
flock -n 9
touch "$ledger"

submit() {
    local stage=$1 previous response
    shift
    previous=$(awk -F '\t' -v key="$stage" '$1==key {print $2}' "$ledger")
    if [[ -n "$previous" ]]; then
        printf '%s\n' "$previous"
        return
    fi
    response=$(sbatch --parsable "$@" --export=ALL slurm/67_breakout_bart_ablation.sbatch "$stage")
    response=${response%%;*}
    [[ "$response" =~ ^[0-9]+$ ]]
    printf '%s\t%s\n' "$stage" "$response" >> "$ledger"
    printf '%s\n' "$response"
}

log="$RRL_BART_CONTROL/slurm"
case "${2:-ablation}" in
  ablation)
    preflight=$(submit preflight --time=00:50:00 --output="$log/preflight-%j.out")
    pilot=$(submit pilot --time=08:00:00 --array=0-11%2 --dependency="afterok:$preflight" --output="$log/pilot-%A_%a.out")
    ;;
  cadence)
    export RRL_BART_CONFIG="$RRL_ROOT/configs/breakout_bart_cadence_500k_20261112.json"
    preflight=$(submit prepare --time=00:30:00 --output="$log/prepare-%j.out")
    pilot=$(submit paired --time=16:00:00 --array=0-9%2 --dependency="afterok:$preflight" --output="$log/paired-%A_%a.out")
    ;;
  stability)
    export RRL_BART_CONFIG="$RRL_ROOT/configs/breakout_bart_stability_1m_20261114.json"
    export RRL_BART_RUNNER=experiments.breakout_bart_stability
    export RRL_BART_HISTORY="${RRL_BART_REFERENCE:-$HOME/projects/rrl_breakout_bart_cadence_500k_20261112_01/pilot/interval_1024}"
    preflight=$(submit stability_prepare --time=01:00:00 --output="$log/prepare-%j.out")
    pilot=$(submit stability_paired --time=24:00:00 --array=0-9%2 --dependency="afterok:$preflight" --output="$log/paired-%A_%a.out")
    ;;
  *)
    echo "Unknown campaign mode: $2" >&2
    exit 2
    ;;
esac
submit report --time=00:15:00 --dependency="afterok:$pilot" --output="$log/report-%j.out" >/dev/null
cat "$ledger"
