#!/usr/bin/env bash
# Finite submission ledger; source snapshot and output directory are distinct.
set -euo pipefail
: "${RRL_ROOT:?Set the frozen source snapshot}"
export RRL_IMPROVEMENT_CONTROL="${RRL_IMPROVEMENT_CONTROL:-$HOME/archive/rrl_improvements_20261108_01}"
export RRL_CONDA_ENV="${RRL_CONDA_ENV:-$HOME/archive/conda_envs/rrl-pong-analysis}"
export RRL_IMPROVEMENT_HISTORICAL="${RRL_IMPROVEMENT_HISTORICAL:-$HOME/archive/rrl_bayesian_breakout_1m_20261028/corrected_common}"
export RRL_IMPROVEMENT_ANOVA="${RRL_IMPROVEMENT_ANOVA:-$HOME/archive/rrl_bayesian_anova_20261102/results/corrected_common}"
mkdir -p "$RRL_IMPROVEMENT_CONTROL/logs"
cd "$RRL_ROOT"
exec 8>"$RRL_IMPROVEMENT_CONTROL/.submission.lock"
flock -n 8 || exit 2
ledger="$RRL_IMPROVEMENT_CONTROL/jobs.tsv";touch "$ledger"
submit() {
    local key=$1 old id;shift
    old=$(awk -v key="$key" '$1==key {print $2}' "$ledger")
    if [[ -n "$old" ]];then printf '%s' "$old";return;fi
    id=$(sbatch --parsable --export=ALL "$@") || return
    id=${id%%;*};[[ "$id" =~ ^[0-9]+$ ]] || return 2
    printf '%s\t%s\n' "$key" "$id" >> "$ledger";printf '%s' "$id"
}
dependency() {
    local condition=$1 id=$2 states state
    if scontrol show job "$id" >/dev/null 2>&1; then
        printf '%s:%s' "$condition" "$id"
        return
    fi
    # Completed jobs disappear from the controller after MinJobAge. Only omit
    # a dependency after accounting proves that all its allocations finished.
    states=$(sacct -X -n -j "$id" --format=State%40 | awk 'NF {print $1}')
    [[ -n "$states" ]] || { printf 'Cannot verify retired dependency %s\n' "$id" >&2;return 2; }
    for state in $states; do
        if [[ "$condition" == afterok ]];then
            [[ "$state" == COMPLETED ]] || { printf 'Dependency %s did not succeed: %s\n' "$id" "$state" >&2;return 2; }
        else
            case "$state" in
                COMPLETED|FAILED|CANCELLED*|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|BOOT_FAIL|PREEMPTED|DEADLINE) ;;
                *) printf 'Dependency %s is not proven terminal: %s\n' "$id" "$state" >&2;return 2 ;;
            esac
        fi
    done
}
prepare=$(submit prepare --output="$RRL_IMPROVEMENT_CONTROL/logs/prepare-%j.out" \
    "$RRL_ROOT/slurm/54_prepare_rrtl_improvements.sbatch")
if [[ "${1:-}" != '--prepare-only' ]];then
    dependencies=();args=()
    d=$(dependency afterok "$prepare");[[ -z "$d" ]] || dependencies+=("$d")
    guard=$(awk '$1=="guard_smoke" {print $2}' "$ledger")
    if [[ -n "$guard" ]];then
        d=$(dependency afterok "$guard");[[ -z "$d" ]] || dependencies+=("$d")
    fi
    if (( ${#dependencies[@]} ));then
        combined=$(IFS=,;printf '%s' "${dependencies[*]}");args=(--dependency="$combined")
    fi
    train=$(submit continuation --array=0-9%2 "${args[@]}" \
        --output="$RRL_IMPROVEMENT_CONTROL/logs/continue-%A_%a.out" \
        "$RRL_ROOT/slurm/55_rrtl_improvement_worker.sbatch" continue)
    d=$(dependency afterany "$train");args=();[[ -z "$d" ]] || args=(--dependency="$d")
    report=$(submit report "${args[@]}" \
        --output="$RRL_IMPROVEMENT_CONTROL/logs/report-%j.out" \
        "$RRL_ROOT/slurm/55_rrtl_improvement_worker.sbatch" report)
    d=$(dependency afterok "$report");args=();[[ -z "$d" ]] || args=(--dependency="$d")
    submit analysis "${args[@]}" --output="$RRL_IMPROVEMENT_CONTROL/logs/analysis-%j.out" \
        "$RRL_ROOT/slurm/57_analyze_rrtl_improvements.sbatch" >/dev/null
fi
cat "$ledger"
