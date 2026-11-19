#!/usr/bin/env bash
# Finite, idempotent DAG. No daemon and no modification of other projects.
set -euo pipefail
: "${RRL_ROOT:?Isolated source snapshot required}"
export RRL_REVISION_CONTROL="${RRL_REVISION_CONTROL:-$HOME/archive/rrl_revision_20261106_01}"
export RRL_CONDA_ENV="${RRL_CONDA_ENV:-$HOME/archive/conda_envs/rrl-pong-analysis}"
export RRL_REVISION_HISTORICAL="${RRL_REVISION_HISTORICAL:-$HOME/archive/rrl_bayesian_breakout_1m_20261028/corrected_common}"
export RRL_REVISION_ANOVA="${RRL_REVISION_ANOVA:-$HOME/archive/rrl_bayesian_anova_20261102/results/corrected_common}"
mkdir -p "$RRL_REVISION_CONTROL/logs"
cd "$RRL_ROOT"
exec 8>"$RRL_REVISION_CONTROL/.submission.lock"
flock -n 8 || exit 2
ledger="$RRL_REVISION_CONTROL/jobs.tsv";touch "$ledger"
submit() {
    local key=$1 old id;shift
    old=$(awk -v key="$key" '$1==key {print $2}' "$ledger")
    if [[ -n "$old" ]];then printf '%s' "$old";return;fi
    id=$(sbatch --parsable --export=ALL "$@") || return
    id=${id%%;*};[[ "$id" =~ ^[0-9]+$ ]] || return 2
    printf '%s\t%s\n' "$key" "$id" >> "$ledger"
    printf '%s' "$id"
}
prepare=$(submit prepare --output="$RRL_REVISION_CONTROL/logs/prepare-%j.out" \
    "$RRL_ROOT/slurm/47_prepare_rrtl_revision.sbatch")
if [[ "${1:-}" != '--prepare-only' ]];then
    frozen=$(submit frozen --array=0-5%1 --dependency="afterok:$prepare" \
        --output="$RRL_REVISION_CONTROL/logs/frozen-%A_%a.out" \
        "$RRL_ROOT/slurm/48_rrtl_revision_worker.sbatch" frozen)
    continuation=$(submit continuation --array=0-5%2 --dependency="afterok:$prepare" \
        --output="$RRL_REVISION_CONTROL/logs/continue-%A_%a.out" \
        "$RRL_ROOT/slurm/48_rrtl_revision_worker.sbatch" continue)
    submit report --dependency="afterany:$frozen:$continuation" \
        --output="$RRL_REVISION_CONTROL/logs/report-%j.out" \
        "$RRL_ROOT/slurm/48_rrtl_revision_worker.sbatch" report >/dev/null
fi
cat "$ledger"
