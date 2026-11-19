#!/usr/bin/env bash
set -euo pipefail
export RRL_ROOT="${RRL_ROOT:?}"
export RRL_FOLLOWUP="${RRL_FOLLOWUP:?}"
export RRL_FOLLOWUP_REFERENCE="${RRL_FOLLOWUP_REFERENCE:?}"
export RRL_CONDA_ENV="${RRL_CONDA_ENV:-$HOME/archive/conda_envs/rrl-pong-analysis}"
cd "$RRL_ROOT"
mkdir -p "$RRL_FOLLOWUP/logs"
exec 8>"$RRL_FOLLOWUP/.submit.lock"
flock -n 8
ledger="$RRL_FOLLOWUP/jobs.tsv"
touch "$ledger"
submit() {
    local stage=$1 old id
    shift
    old=$(awk -v stage="$stage" '$1==stage && NF==2 {print $2}' "$ledger")
    if [[ -n "$old" ]]; then
        printf '%s' "$old"
        return
    fi
    local suffix='%j'
    if [[ "$stage" == factorial || "$stage" == refit || "$stage" == double || "$stage" == diagnostics ]]; then
        suffix='%A_%a'
    fi
    if ! id=$(sbatch --parsable --export=ALL --job-name="rrl-$stage" \
        --output="$RRL_FOLLOWUP/logs/$stage-$suffix.out" --error="$RRL_FOLLOWUP/logs/$stage-$suffix.err" \
        "$@" "$RRL_ROOT/slurm/69_rrtl_followup.sbatch" "$stage"); then
        return 1
    fi
    id=${id%%;*}
    [[ "$id" =~ ^[0-9]+$ ]] || return 1
    printf '%s\t%s\n' "$stage" "$id" >> "$ledger"
    printf '%s' "$id"
}
prepare=$(submit prepare --time=02:00:00)
factorial=$(submit factorial --dependency="afterok:$prepare" --array=0-4%1)
refit=$(submit refit --dependency="afterok:$prepare" --array=0-4%1)
double=$(submit double --dependency="afterok:$prepare" --array=0-4%1 --time=02:00:00)
diagnostics=$(submit diagnostics --dependency="afterok:$prepare" --array=0-4%1)
submit report --dependency="afterok:$factorial:$refit:$double:$diagnostics" --time=02:00:00 >/dev/null
awk 'NF==2 && $1!="worker"' "$ledger"
