#!/usr/bin/env bash
set -euo pipefail
: "${RRL_ROOT:?Isolated ANOVA source snapshot required}"
export RRL_CONDA_ENV="${RRL_CONDA_ENV:-$HOME/archive/conda_envs/rrl-pong-analysis}"
export RRL_CONTROL="${RRL_CONTROL:-$HOME/archive/rrl_bayesian_anova_20261102}"
export RRL_BREAK_1M_ROOT="$RRL_CONTROL/results"
mkdir -p "$RRL_CONTROL/logs"
cd "$RRL_ROOT"
exec 8>"$RRL_CONTROL/.submission.lock";flock -n 8 || exit 2
ledger="$RRL_CONTROL/jobs.tsv";touch "$ledger"
submit() {
    local key=$1;shift
    local old id
    old=$(awk -v k="$key" '$1==k {print $2}' "$ledger")
    if [[ -n "$old" ]];then printf '%s' "$old";return;fi
    id=$(sbatch --parsable --export=ALL "$@") || return
    id=${id%%;*};[[ "$id" =~ ^[0-9]+$ ]] || return 2
    printf '%s\t%s\n' "$key" "$id" >> "$ledger";printf '%s' "$id"
}
prepare=$(submit prepare --output="$RRL_CONTROL/logs/prepare-%j.out" "$RRL_CONTROL/slurm/45_prepare_breakout_anova.sbatch")
train=$(submit train --dependency="afterok:$prepare" --array=0-4%2 \
    --job-name=rrl-breakout-anova --output="$RRL_CONTROL/logs/train-%A_%a.out" \
    "$RRL_CONTROL/slurm/38_breakout_1m_group.sbatch" corrected_common)
submit calendar --dependency="afterany:$train" --output="$RRL_CONTROL/logs/calendar-%j.out" \
    "$RRL_CONTROL/slurm/42_close_breakout_calendar.sbatch" corrected_common >/dev/null
cat "$ledger"
