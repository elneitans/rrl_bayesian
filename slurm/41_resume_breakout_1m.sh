#!/usr/bin/env bash
# One explicit, finite recovery; no polling daemon or recurring submissions.
# Usage: SCRIPT METHOD GROUP PREVIOUS_ARRAY_JOB_ID
set -euo pipefail
[[ $# == 3 && "$1" =~ ^(corrected_common|cart|extra_trees|bart)$ && "$2" =~ ^[0-4]$ && "$3" =~ ^[0-9]+$ ]] || exit 2
method=$1; group=$2; previous=$3
export RRL_ROOT="${RRL_ROOT:-$HOME/projects/rrl_bayesian}"
export RRL_CONDA_ENV="${RRL_CONDA_ENV:-$HOME/archive/conda_envs/rrl-pong-analysis}"
export RRL_CONTROL="${RRL_CONTROL:-$HOME/archive/rrl_bayesian_control_20261027}"
export RRL_BREAK_1M_ROOT="${RRL_BREAK_1M_ROOT:-$HOME/archive/rrl_bayesian_breakout_1m_20261028}"
cd "$RRL_ROOT"
exec 8>"$RRL_CONTROL/.submit-breakout-1m.lock"
flock -n 8 || exit 2
ledger="$RRL_CONTROL/breakout_1m_recovery_jobs.tsv"
touch "$ledger"
original=$(awk -v k="${method}_train" '$1==k {print $2}' "$RRL_CONTROL/breakout_1m_jobs.tsv")
if [[ "$previous" != "$original" ]]; then
  known=$(awk -v m="$method" -v g="$group" -v j="$previous" '$1==m && $2==g && $4==j {print $4}' "$ledger")
  [[ "$known" == "$previous" ]] || { echo 'Previous job is not a known worker for this method/group' >&2; exit 2; }
fi
existing=$(awk -v m="$method" -v g="$group" -v j="$previous" '$1==m && $2==g && $3==j {print $4}' "$ledger")
if [[ -n "$existing" ]]; then
  echo "Already submitted recovery: $existing (inspect it before another recovery)"
  exit 0
fi
record=$(sacct -j "$previous" --noheader --parsable2 --format=JobID%40,State,ExitCode | \
  awk -F'|' -v id="${previous}_${group}" '{gsub(/^ +| +$/,"",$1);if($1==id)print $2 "|" $3}')
case "$record" in
  'COMPLETED|0:0') echo 'Group already completed; nothing submitted'; exit 0 ;;
  'FAILED|3:0') ;;
  *) echo "No recovery submitted: previous group state is '$record' (only FAILED/3:0 is eligible)"; exit 3 ;;
esac
job=$(sbatch --parsable --export=ALL --array="$group%1" \
  --job-name="rrl-break-1m-resume-$method" \
  --output="$RRL_CONTROL/logs/break-1m-resume-$method-%A_%a.out" \
  "$RRL_CONTROL/slurm/38_breakout_1m_group.sbatch" "$method")
job=${job%%;*}
[[ "$job" =~ ^[0-9]+$ ]] || exit 2
# Record the successful submission before optional reporting: a later failure
# must not cause duplicate training on a repeated invocation.
printf '%s\t%s\t%s\t%s\n' "$method" "$group" "$previous" "$job" >> "$ledger"
echo "Recovery submitted: method=$method group=$group previous=$previous job=$job"
echo 'After recovery, inspect states and submit the existing report script 39 when appropriate.'
