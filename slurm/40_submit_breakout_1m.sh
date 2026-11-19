#!/usr/bin/env bash
# Login-node safe: shell and Slurm only; all Python runs on compute nodes.
set -euo pipefail
export RRL_ROOT="${RRL_ROOT:-$HOME/projects/rrl_bayesian}"
export RRL_CONDA_ENV="${RRL_CONDA_ENV:-$HOME/archive/conda_envs/rrl-pong-analysis}"
export RRL_CONTROL="${RRL_CONTROL:-$HOME/archive/rrl_bayesian_control_20261027}"
export RRL_BREAK_1M_ROOT="${RRL_BREAK_1M_ROOT:-$HOME/archive/rrl_bayesian_breakout_1m_20261028}"
mkdir -p "$RRL_BREAK_1M_ROOT" "$RRL_CONTROL/logs"
cd "$RRL_ROOT"
exec 8>"$RRL_CONTROL/.submit-breakout-1m.lock"
flock -n 8 || exit 2
ledger="$RRL_CONTROL/breakout_1m_jobs.tsv"
touch "$ledger"
submit() {
  local key=$1; shift
  local previous
  previous=$(awk -v k="$key" '$1==k {print $2}' "$ledger")
  if [[ -n "$previous" ]]; then
    printf '%s' "$previous"; return
  fi
  local id
  id=$(sbatch --parsable --export=ALL "$@") || return
  id=${id%%;*}
  [[ "$id" =~ ^[0-9]+$ ]] || { echo 'Invalid sbatch response' >&2; return 2; }
  printf '%s\t%s\n' "$key" "$id" >> "$ledger"
  printf '%s' "$id"
}
# Resume safely within the same submission window. A terminal failed/cancelled
# preparation must be reviewed explicitly rather than silently reused.
prep=$(submit prepare --output="$RRL_CONTROL/logs/break-1m-prep-%j.out" "$RRL_CONTROL/slurm/37_prepare_breakout_1m.sbatch")
for method in corrected_common cart extra_trees bart; do
  train=$(submit "${method}_train" --dependency="afterok:$prep" --array=0-4%1 \
    --job-name="rrl-break-1m-$method" --output="$RRL_CONTROL/logs/break-1m-$method-%A_%a.out" \
    "$RRL_CONTROL/slurm/38_breakout_1m_group.sbatch" "$method")
  submit "${method}_report" --dependency="afterany:$train" \
    --output="$RRL_CONTROL/logs/break-1m-report-$method-%j.out" \
    "$RRL_CONTROL/slurm/39_breakout_1m_report.sbatch" "$method" >/dev/null
done
cat "$ledger"
