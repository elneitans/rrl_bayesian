#!/usr/bin/env bash
# Idempotent finite submission of the remaining two declared conditions.
set -euo pipefail
export RRL_ROOT="${RRL_ROOT:-$HOME/projects/rrl_bayesian}"
export RRL_CONDA_ENV="${RRL_CONDA_ENV:-$HOME/archive/conda_envs/rrl-pong-analysis}"
control="${RRL_CONTROL_ROOT:-$HOME/archive/rrl_bayesian_control_20261027}"
export RRL_CONTROL_SCRIPTS="$control/slurm"
cd "$RRL_ROOT"
exec 9>"$control/.submission.lock"
flock -n 9 || exit 2
training="$control/long_training_jobs.tsv"
evaluations="$control/long_evaluation_jobs.tsv"
dependency_args() {
  local mode=$1 job=$2 final
  dependency=()
  if [[ -n "$(squeue -h -j "$job" -o '%i' 2>/dev/null || true)" ]]; then
    dependency=("--dependency=$mode:$job")
    return
  fi
  final=$(sacct -n -X -P -j "$job" --format=State | head -n 1)
  case "$final" in
    COMPLETED) return ;;
    FAILED|TIMEOUT|CANCELLED*|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED)
      [[ "$mode" == afterany ]] && return ;;
  esac
  echo "Cannot satisfy $mode dependency on job $job (state=$final)" >&2
  exit 2
}
for method in cart bart; do
  prepare=$(awk -F '\t' -v label="breakout_$method" '$1==label {print $2}' "$control/long_prepare_jobs.tsv")
  [[ "$prepare" =~ ^[0-9]+$ ]] || exit 2
  campaign="$HOME/archive/rrl_bayesian_long_runs_20261027/breakout_${method}_2m_20261027"
  train=$(awk -F '\t' -v label="breakout_${method}_attempt1" '$1==label {print $2}' "$training")
  if [[ -z "$train" ]]; then
    count=$(squeue -h -r -u "$USER" -o '%i' | wc -l)
    (( count+1<=32 )) || { echo 'Quota pause; rerun this script for missing labels'; exit 75; }
    export RRL_LOG_DIR="$control/environment_long/breakout_${method}_attempt1"
    dependency_args afterok "$prepare"
    train=$(sbatch --parsable ${dependency[@]+"${dependency[@]}"} --job-name="rrl-break-${method}-2m" \
      --output="$control/logs/breakout-${method}-2m-%j.out" "$control/slurm/29_long_condition.sbatch" "$campaign")
    train=${train%%;*}; [[ "$train" =~ ^[0-9]+$ ]] || exit 2
    printf 'breakout_%s_attempt1\t%s\n' "$method" "$train" >> "$training"
  fi
  evaluation=$(awk -F '\t' -v label="breakout_${method}_evaluation_attempt1" '$1==label {print $2}' "$evaluations")
  if [[ -z "$evaluation" ]]; then
    count=$(squeue -h -r -u "$USER" -o '%i' | wc -l)
    (( count+1<=32 )) || { echo 'Quota pause; rerun this script for missing labels'; exit 75; }
    export RRL_LOG_DIR="$control/environment_evaluation/breakout_${method}_attempt1"
    dependency_args afterany "$train"
    evaluation=$(sbatch --parsable ${dependency[@]+"${dependency[@]}"} \
      --output="$control/logs/breakout-${method}-evaluation-%j.out" "$control/slurm/30_evaluate_condition.sbatch" "$campaign")
    evaluation=${evaluation%%;*}; [[ "$evaluation" =~ ^[0-9]+$ ]] || exit 2
    printf 'breakout_%s_evaluation_attempt1\t%s\n' "$method" "$evaluation" >> "$evaluations"
  fi
  printf '%s train=%s evaluate=%s\n' "$method" "$train" "$evaluation"
done
