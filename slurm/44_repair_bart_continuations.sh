#!/usr/bin/env bash
# One-time repair of v1 gates: compute-node sacct cannot reach dwemer.
# Run on the login node; Python/model work remains in submitted compute jobs.
set -euo pipefail
export RRL_ROOT="${RRL_ROOT:-$HOME/projects/rrl_bayesian}"
export RRL_CONDA_ENV="${RRL_CONDA_ENV:-$HOME/archive/conda_envs/rrl-pong-analysis}"
export RRL_CONTROL="${RRL_CONTROL:-$HOME/archive/rrl_bayesian_control_20261027}"
export RRL_BREAK_1M_ROOT="${RRL_BREAK_1M_ROOT:-$HOME/archive/rrl_bayesian_breakout_1m_20261028}"
cd "$RRL_ROOT"
exec 8>"$RRL_CONTROL/.submit-breakout-1m.lock"
flock -n 8 || exit 2
original=$(sacct -j 35470 --noheader --parsable2 --format=JobID%40,State,ExitCode | awk -F'|' '$1=="35470_0" {print $2 "|" $3}')
[[ "$original" == 'FAILED|3:0' ]] || { echo "Original group0 requires review: $original"; exit 2; }
grep -q 'sacct: error' "$RRL_CONTROL/logs/break-1m-bart-continuation-35728_0.out"
# Replace only obsolete, non-running copies of the old gate. Preserve running
# tasks if a state transition races this repair and stop for review.
for job in 35729 35730 35731 35732; do
  state=$(squeue -h -j "$job" -o '%T')
  if [[ "$state" == PENDING ]]; then
    scancel --state=PENDING "$job"
    state=$(squeue -h -j "$job" -o '%T')
  fi
  [[ -z "$state" ]] || { echo "Old gate $job is $state; not replaced"; exit 3; }
done
ledger="$RRL_CONTROL/breakout_1m_recovery_v2_jobs.tsv"
touch "$ledger"
for group in 0 1 2 3 4; do
  existing=$(awk -v g="$group" '$1=="bart" && $2==g {print $4}' "$ledger")
  if [[ -n "$existing" ]]; then echo "Already submitted $group=$existing"; continue; fi
  dependency=()
  # Group0 is already terminal, verified above; do not attach an expired ID.
  if (( group>0 )); then dependency=(--dependency="afterany:35470_$group"); fi
  id=$(sbatch --parsable --export=ALL --array="$group%1" \
    ${dependency[@]+"${dependency[@]}"} --job-name=rrl-break-bart-resume-v2 \
    --output="$RRL_CONTROL/logs/break-1m-bart-resume-v2-%A_%a.out" \
    "$RRL_CONTROL/slurm/43_gated_breakout_resume.sbatch" bart 35470)
  id=${id%%;*}; [[ "$id" =~ ^[0-9]+$ ]] || exit 2
  printf 'bart\t%s\t35470\t%s\n' "$group" "$id" >> "$ledger"
  printf 'bart\t%s\t35470\t%s\n' "$group" "$id" >> "$RRL_CONTROL/breakout_1m_recovery_jobs.tsv"
done
ids=$(awk '{printf ":%s",$4}' "$ledger")
for report in 35471 35695; do
  state=$(squeue -h -j "$report" -o '%T')
  [[ "$state" == PENDING ]] || { echo "Report $report is $state; review dependency"; exit 3; }
  scontrol update JobId="$report" Dependency="afterany:35470$ids"
done
cat "$ledger"
