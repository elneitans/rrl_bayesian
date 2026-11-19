#!/usr/bin/env bash
set -euo pipefail
export RRL_BART_COVERAGE="${1:?New campaign path required}"
export RRL_ROOT="$RRL_BART_COVERAGE/code"
export RRL_BART_CADENCE="$HOME/projects/rrl_breakout_bart_cadence_500k_20261112_01"
export RRL_BART_STABILITY="$HOME/projects/rrl_breakout_bart_stability_1m_20261114_01"
export RRL_CONDA_ENV="$HOME/archive/conda_envs/rrl-pong-analysis"
mkdir -p "$RRL_BART_COVERAGE/slurm"
ledger="$RRL_BART_COVERAGE/jobs.tsv"
touch "$ledger"
exec 8>"$RRL_BART_COVERAGE/submission.lock"
flock -n 8
submit() {
    local stage=$1 prior id
    shift
    prior=$(awk -v stage="$stage" '$1==stage && NF==2 {print $2}' "$ledger")
    if [[ -n "$prior" ]]; then
        printf '%s' "$prior"
        return
    fi
    if ! id=$(sbatch --parsable --export=ALL --job-name="bart-$stage" \
        --output="$RRL_BART_COVERAGE/slurm/$stage-%A_%a.out" "$@" \
        "$RRL_ROOT/slurm/71_bart_replay_coverage.sbatch" "$stage"); then
        return 1
    fi
    id=${id%%;*}
    [[ "$id" =~ ^[0-9]+$ ]] || return 1
    printf '%s\t%s\n' "$stage" "$id" >> "$ledger"
    printf '%s' "$id"
}
prepare=$(submit prepare --time=01:00:00)
coverage=$(submit coverage --dependency="afterok:$prepare" --array=0-4%1 --time=08:00:00)
control=$(submit control --dependency="afterok:$prepare" --array=0-4%2 --time=24:00:00)
submit report --dependency="afterok:$coverage:$control" --time=01:00:00 >/dev/null
cat "$ledger"
