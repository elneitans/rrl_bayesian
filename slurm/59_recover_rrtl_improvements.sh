#!/usr/bin/env bash
# Resume the unchanged campaign from a verified copy in the writable home.
set -euo pipefail
export RRL_ROOT="${RRL_ROOT:-$HOME/archive/rrl_improvement_code_20261108_v1}"
export RRL_IMPROVEMENT_CONTROL="${RRL_IMPROVEMENT_CONTROL:-$HOME/projects/rrl_evaluation_recovery_20261109_02}"
export RRL_CONDA_ENV="${RRL_CONDA_ENV:-$HOME/archive/conda_envs/rrl-pong-analysis}"
[[ -f "$RRL_IMPROVEMENT_CONTROL/RECOVERY_COPY_VERIFIED.txt" ]] || { echo 'Verified recovery copy required' >&2; exit 2; }
cd "$RRL_ROOT"
mkdir -p "$RRL_IMPROVEMENT_CONTROL/logs"
exec 8>"$RRL_IMPROVEMENT_CONTROL/.recovery_submission.lock"
flock -n 8 || exit 2
ledger="$RRL_IMPROVEMENT_CONTROL/recovery_jobs.tsv"
touch "$ledger"
submit() {
    local key=$1 old id
    shift
    old=$(awk -v key="$key" '$1==key {print $2}' "$ledger")
    if [[ -n "$old" ]]; then printf '%s' "$old"; return; fi
    id=$(sbatch --parsable --export=ALL "$@") || return
    id=${id%%;*}
    [[ "$id" =~ ^[0-9]+$ ]] || return 2
    printf '%s\t%s\n' "$key" "$id" >> "$ledger"
    printf '%s' "$id"
}
preflight=$(submit preflight \
    --output="$RRL_IMPROVEMENT_CONTROL/logs/recovery-preflight-%j.out" \
    --error="$RRL_IMPROVEMENT_CONTROL/logs/recovery-preflight-%j.err" \
    "$RRL_IMPROVEMENT_CONTROL/60_preflight_rrtl_recovery.sbatch")
train=$(submit resume --dependency="afterok:$preflight" --array=8-9%2 \
    --output="$RRL_IMPROVEMENT_CONTROL/logs/recovery-%A_%a.out" \
    --error="$RRL_IMPROVEMENT_CONTROL/logs/recovery-%A_%a.err" \
    "$RRL_IMPROVEMENT_CONTROL/61_resume_rrtl_incomplete.sbatch")
report=$(submit report --dependency="afterany:$train" \
    --output="$RRL_IMPROVEMENT_CONTROL/logs/recovery-report-%j.out" \
    --error="$RRL_IMPROVEMENT_CONTROL/logs/recovery-report-%j.err" \
    "$RRL_ROOT/slurm/55_rrtl_improvement_worker.sbatch" report)
submit analysis --dependency="afterok:$report" \
    --output="$RRL_IMPROVEMENT_CONTROL/logs/recovery-analysis-%j.out" \
    --error="$RRL_IMPROVEMENT_CONTROL/logs/recovery-analysis-%j.err" \
    "$RRL_ROOT/slurm/57_analyze_rrtl_improvements.sbatch" >/dev/null
submit guard_final --dependency="afterok:$report" \
    --output="$RRL_IMPROVEMENT_CONTROL/logs/recovery-guard-%j.out" \
    --error="$RRL_IMPROVEMENT_CONTROL/logs/recovery-guard-%j.err" \
    "$RRL_IMPROVEMENT_CONTROL/58_audit_rrtl_improvement_guards.sbatch" campaign >/dev/null
cat "$ledger"
