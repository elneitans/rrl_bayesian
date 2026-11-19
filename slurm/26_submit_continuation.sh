#!/usr/bin/env bash
# Missing-label continuation only; failed IDs are audited before retrying.
set -euo pipefail
export RRL_ROOT="${RRL_ROOT:-$HOME/projects/rrl_bayesian}"
export RRL_CONDA_ENV="${RRL_CONDA_ENV:-$HOME/archive/conda_envs/rrl-pong-analysis}"
export RRL_CALIBRATION_RESULTS="$HOME/archive/rrl_bayesian_calibrations_20261027"
cd "$RRL_ROOT"
if [[ -s "$HOME/archive/rrl_bayesian_control_20261027/baseline_finish_20261028.tsv" ]]; then
  echo 'Baseline continuation already recorded by 35_submit_baseline_finish.sh; inspect its ledger instead.'
  exit 2
fi
[[ -L results ]] || { echo 'Archive migration must finish first'; exit 2; }
record="$RRL_ROOT/results/submissions/all_algorithms_2m_20261027"
mkdir -p "$record"
exec 9>"$record/.lock"
flock -n 9 || exit 2
touch "$record/jobs.tsv"
# Calibrations may have been submitted independently while results migrated.
calibration_record="$HOME/archive/rrl_bayesian_control_20261027/calibration_jobs.tsv"
if [[ -f "$calibration_record" ]]; then
  while IFS=$'\t' read -r label id; do
    [[ "$label" =~ ^(prepare_calibrations|calibrations)$ && "$id" =~ ^[0-9]+$ ]] || exit 2
    existing=$(awk -F '\t' -v label="$label" '$1==label {print $2}' "$record/jobs.tsv")
    if [[ -n "$existing" ]]; then [[ "$existing" == "$id" ]] || exit 2
    else printf '%s\t%s\n' "$label" "$id" >> "$record/jobs.tsv"
    fi
  done < "$calibration_record"
fi
submit() {
  local label=$1 slots=$2 id old count
  shift 2
  old=$(awk -F '\t' -v label="$label" '$1==label {print $2}' "$record/jobs.tsv")
  if [[ -n "$old" ]]; then echo "$old"; return; fi
  count=$(squeue -h -r -u "$USER" -o '%i' | wc -l)
  (( count+slots <= 32 )) || { echo 'Quota pause; rerun to submit missing labels only.' >&2; exit 75; }
  id=$(sbatch --parsable "$@"); id=${id%%;*}
  [[ "$id" =~ ^[0-9]+$ ]] || exit 2
  printf '%s\t%s\n' "$label" "$id" >> "$record/jobs.tsv"
  echo "$id"
}
auth=$(submit authorize_breakout_2m 1 --export=ALL,RRL_TASK=authorize-breakout slurm/23_continue_existing.sbatch)
submit pong_recovery 7 --array=5-11%1 --export=ALL,RRL_TASK=pong slurm/23_continue_existing.sbatch >/dev/null
submit breakout_baseline_2m 10 --array=0-9%1 --dependency="afterok:$auth" \
  --export=ALL,RRL_TASK=breakout slurm/23_continue_existing.sbatch >/dev/null
submit breakout_cart_evaluation_repair 1 --export=ALL,RRL_TASK=breakout-evaluation-repair \
  slurm/23_continue_existing.sbatch >/dev/null
prep=$(submit prepare_calibrations 1 slurm/24_prepare_calibrations.sbatch)
submit calibrations 6 --array=0-5%1 --dependency="afterok:$prep" slurm/25_calibration.sbatch >/dev/null
cat "$record/jobs.tsv"
echo 'Next: audit results, collect compatible costs, authorize and submit long CART/ExtraTrees/BART campaigns to 2M.'
