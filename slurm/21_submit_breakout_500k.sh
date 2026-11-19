#!/usr/bin/env bash
# Submit from the login node. --resume submits only missing labels, never retries failed IDs.
set -euo pipefail
resume=0
if [[ "${1:-}" == --resume ]]; then resume=1; shift; fi
(( $# == 0 )) || { echo 'Usage: bash slurm/21_submit_breakout_500k.sh [--resume]'; exit 2; }
export RRL_ROOT="${RRL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$RRL_ROOT"
export RRL_CONDA_ENV="${RRL_CONDA_ENV:-/home/natan.brugueras/archive/conda_envs/rrl-pong-analysis}"
export RRL_BREAKOUT_BASE="$RRL_ROOT/results/breakout_baseline_15_linux_20261026_01"
export RRL_BREAKOUT_CART="$RRL_ROOT/results/breakout_cart_10seeds_500k_20261026_01"
record="$RRL_ROOT/results/submissions/breakout_500k_10seeds_20261026_01"
[[ -f "$RRL_BREAKOUT_BASE/campaign.json" ]] || { echo 'Missing existing baseline campaign'; exit 2; }
for file in slurm/19_breakout_500k_prepare.sbatch slurm/20_breakout_cell_group.sbatch slurm/12_online_report.sbatch; do
  [[ -f "$file" ]] || exit 2
done
if (( resume )); then [[ -d "$record" ]] || exit 2
else
  mkdir -p "$(dirname "$record")"
  mkdir "$record" || { echo 'Submission exists; inspect jobs.tsv and use --resume for missing submissions only.'; exit 2; }
fi
exec 9>"$record/.lock"
flock -n 9 || { echo 'Another submitter is active'; exit 2; }
touch "$record/jobs.tsv"
submit() {
  local label="$1" old count id slots=1 arg
  shift
  old=$(awk -F '\t' -v label="$label" '$1==label {print $2}' "$record/jobs.tsv")
  if [[ -n "$old" ]]; then echo "$old"; return; fi
  count=$(squeue -h -r -u "$USER" -o '%i' | wc -l)
  for arg in "$@"; do
    if [[ "$arg" == --array=0-2%1 ]]; then slots=3; fi
  done
  if (( count + slots > 32 )); then
    echo 'Quota pause: inspect jobs.tsv and rerun with --resume after slots free.' >&2
    exit 75
  fi
  id=$(sbatch --parsable "$@")
  id="${id%%;*}"
  [[ "$id" =~ ^[0-9]+$ ]] || { echo 'Unexpected sbatch response' >&2; exit 2; }
  printf '%s\t%s\n' "$label" "$id" >> "$record/jobs.tsv"
  echo "$id"
}
prep=$(submit prepare slurm/19_breakout_500k_prepare.sbatch)
for method in baseline cart; do
  campaign="$RRL_BREAKOUT_BASE"
  [[ "$method" != cart ]] || campaign="$RRL_BREAKOUT_CART"
  train=$(submit "${method}_train" --dependency="afterok:$prep" --array=0-2%1 \
    --time=24:00:00 --output="rrl-breakout-${method}-train-%A_%a.out" \
    slurm/20_breakout_cell_group.sbatch train "$campaign")
  evaluation=$(submit "${method}_evaluate" --dependency="afterok:$prep,afterany:$train" --array=0-2%1 \
    --time=08:00:00 --output="rrl-breakout-${method}-eval-%A_%a.out" \
    slurm/20_breakout_cell_group.sbatch evaluate "$campaign")
  submit "${method}_report" --dependency="afterok:$prep,afterany:$evaluation" \
    --output="rrl-breakout-${method}-report-%j.out" \
    slurm/12_online_report.sbatch --campaign "$campaign" >/dev/null
done
cat "$record/jobs.tsv"
echo 'Only development evaluation at 500k was scheduled. No reserved test.'
