#!/usr/bin/env bash
# Only after checking the successful preflight and storage/cost projection.
set -euo pipefail
: "${RRL_ROOT:?frozen source required}" "${RRL_CONFIRM_CONTROL:?control required}"
export RRL_ROOT RRL_CONFIRM_CONTROL
export RRL_CONDA_ENV="${RRL_CONDA_ENV:-$HOME/archive/conda_envs/rrl-pong-analysis}"
[[ -f "$RRL_CONFIRM_CONTROL/preflight_complete.json" ]] || exit 2
cd "$RRL_ROOT"
exec 8>"$RRL_CONFIRM_CONTROL/.submit.lock";flock -n 8 || exit 2
ledger="$RRL_CONFIRM_CONTROL/jobs.tsv";touch "$ledger"
submit() {
 local key=$1 id old;shift
 old=$(awk -v key="$key" '$1==key {print $2}' "$ledger")
 if [[ -n "$old" ]];then printf '%s' "$old";return;fi
 id=$(sbatch --parsable --export=ALL "$@");id=${id%%;*};[[ "$id" =~ ^[0-9]+$ ]] || return 2
 printf '%s\t%s\n' "$key" "$id" >> "$ledger";printf '%s' "$id"
}
worker="${RRL_CONFIRM_WORKER:-$RRL_CONFIRM_CONTROL/63_confirmation_worker.sbatch}"
[[ -f "$worker" ]] || exit 2
integration_batch="${RRL_CONFIRM_INTEGRATION_BATCH:-$RRL_CONFIRM_CONTROL/65_confirmation_integration.sbatch}"
[[ -f "$integration_batch" ]] || { echo 'Integration batch required before submitting'; exit 2; }
parents=$(submit parents --array=0-4%2 --output="$RRL_CONFIRM_CONTROL/logs/parents-%A_%a.out" --error="$RRL_CONFIRM_CONTROL/logs/parents-%A_%a.err" "$worker" parents)
integration=$(submit integration --dependency="afterok:$parents" --output="$RRL_CONFIRM_CONTROL/logs/integration-%j.out" --error="$RRL_CONFIRM_CONTROL/logs/integration-%j.err" "$integration_batch")
prepare=$(submit prepare --dependency="afterok:$integration" --output="$RRL_CONFIRM_CONTROL/logs/prepare-%j.out" --error="$RRL_CONFIRM_CONTROL/logs/prepare-%j.err" "$worker" prepare)
train=$(submit continue --dependency="afterok:$prepare" --array=0-4%2 --output="$RRL_CONFIRM_CONTROL/logs/continue-%A_%a.out" --error="$RRL_CONFIRM_CONTROL/logs/continue-%A_%a.err" "$worker" continue)
report=$(submit report --dependency="afterany:$train" --output="$RRL_CONFIRM_CONTROL/logs/report-%j.out" --error="$RRL_CONFIRM_CONTROL/logs/report-%j.err" "$worker" report)
for stage in analyze audit;do
 submit "$stage" --dependency="afterok:$report" --output="$RRL_CONFIRM_CONTROL/logs/$stage-%j.out" --error="$RRL_CONFIRM_CONTROL/logs/$stage-%j.err" "$worker" "$stage" >/dev/null
done
cat "$ledger"
