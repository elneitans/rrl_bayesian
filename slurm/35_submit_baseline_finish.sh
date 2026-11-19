#!/usr/bin/env bash
# Finite baseline DAG. Repeating submits only missing labels, never retries failed IDs.
set -euo pipefail
export RRL_ROOT="${RRL_ROOT:-$HOME/projects/rrl_bayesian}"
export RRL_CONDA_ENV="${RRL_CONDA_ENV:-$HOME/archive/conda_envs/rrl-pong-analysis}"
control="${RRL_CONTROL_ROOT:-$HOME/archive/rrl_bayesian_control_20261027}"
export RRL_LOG_DIR="$control/environment_baseline_finish"
record="$control/baseline_finish_20261028.tsv"
cd "$RRL_ROOT"
exec 9>"$control/.baseline_submission.lock"
flock -n 9 || exit 2
touch "$record"
submit() {
  local label=$1 slots=$2 old count id arg clause mode job final joined
  local -a resolved clauses kept
  shift 2
  old=$(awk -F '\t' -v label="$label" '$1==label {print $2}' "$record")
  if [[ -n "$old" ]]; then echo "$old"; return; fi
  count=$(squeue -h -r -u "$USER" -o '%i' | wc -l)
  (( count+slots<=32 )) || { echo 'Quota pause; rerun for missing labels' >&2; exit 75; }
  resolved=()
  for arg in "$@"; do
    if [[ "$arg" == --dependency=* ]]; then
      IFS=',' read -r -a clauses <<< "${arg#--dependency=}"
      kept=()
      for clause in "${clauses[@]}"; do
        mode=${clause%%:*}; job=${clause#*:}
        if [[ -n "$(squeue -h -j "$job" -o '%i' 2>/dev/null || true)" ]]; then
          kept+=("$clause")
        else
          final=$(sacct -n -X -P -j "$job" --format=State | head -n 1)
          case "$final" in
            COMPLETED) ;;
            FAILED|TIMEOUT|CANCELLED*|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED)
              [[ "$mode" == afterany ]] || { echo "Unsatisfied dependency $clause ($final)" >&2; exit 2; } ;;
            *) echo "Unknown dependency $clause ($final)" >&2; exit 2 ;;
          esac
        fi
      done
      if (( ${#kept[@]} )); then
        joined=$(IFS=','; echo "${kept[*]}")
        resolved+=("--dependency=$joined")
      fi
    else resolved+=("$arg")
    fi
  done
  id=$(sbatch --parsable "${resolved[@]}"); id=${id%%;*}
  [[ "$id" =~ ^[0-9]+$ ]] || exit 2
  printf '%s\t%s\n' "$label" "$id" >> "$record"
  echo "$id"
}
# Existing worker is live at creation; this gives the verified copy time to finish.
gate=$(submit storage_gate 1 --dependency="afterany:${RRL_STORAGE_AFTER_JOB:-34956}" \
  --output="$control/logs/baseline-gate-%j.out" "$control/slurm/33_baseline_storage_gate.sbatch")
for game in pong breakout; do
  array=0-1%1; slots=2
  if [[ "$game" == breakout ]]; then array=0-2%1; slots=3; fi
  train=$(submit "${game}_train" "$slots" --dependency="afterok:$gate" --array="$array" \
    --output="$control/logs/baseline-${game}-train-%A_%a.out" "$control/slurm/34_baseline_group.sbatch" train "$game")
  submit "${game}_evaluate_report" 1 --dependency="afterok:$gate,afterany:$train" \
    --output="$control/logs/baseline-${game}-evaluate-%j.out" "$control/slurm/34_baseline_group.sbatch" evaluate "$game" >/dev/null
done
cat "$record"
