#!/usr/bin/env bash
# Run with bash on the login node; this submits jobs, it is not an sbatch itself.
set -euo pipefail
resume=0
export RRL_OCATARI_ONLY=0
while (( $# )); do
  case "$1" in
    --resume) resume=1 ;;
    --ocatari-only) RRL_OCATARI_ONLY=1 ;;
    *) echo "Usage: bash slurm/17_submit_pong_breakout.sh [--resume] [--ocatari-only]" >&2; exit 2 ;;
  esac
  shift
done
analysis_array=0-1%1
if (( RRL_OCATARI_ONLY )); then analysis_array=0-0%1; fi
export RRL_ROOT="${RRL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$RRL_ROOT"
export RRL_CONDA_ENV="${RRL_CONDA_ENV:-/home/natan.brugueras/archive/conda_envs/rrl-pong-analysis}"
export RRL_ORIGINAL_CAMPAIGN="$RRL_ROOT/results/pong_baseline_progressive_linux_20261014_01"
export RRL_VISUAL_EXTRA="$RRL_ROOT/results/pong_visual_2m_extra12_20261025_01"
export RRL_OCATARI_EXTRA="$RRL_ROOT/results/pong_ocatari_2m_extra12_20261026_01"
export RRL_BREAKOUT_CAMPAIGN="$RRL_ROOT/results/breakout_baseline_15_linux_20261026_01"
export RRL_ANALYSIS_OUTPUT="$RRL_ROOT/results/pong_analysis_v2_linux_20261025_01"
export RRL_ANALYSIS_CODE="$RRL_ROOT/historical/pong_pilot_linux_01_code"
RRL_SUBMISSION_DIR="${RRL_SUBMISSION_DIR:-$RRL_ROOT/results/submissions/pong_breakout_20261026_01}"

for path in configs/pong_online_baseline_visual_2m_extra12.json \
  configs/pong_online_baseline_ocatari_2m_extra12.json \
  configs/breakout_online_baseline_15seeds.json \
  slurm/common.sh slurm/09_online_train_cell.sbatch \
  slurm/10_online_evaluate.sbatch slurm/11_online_resume.sbatch \
  slurm/12_online_report.sbatch slurm/18_evaluate_cells.sbatch \
  "$RRL_ORIGINAL_CAMPAIGN/campaign.json" \
  "$RRL_VISUAL_EXTRA/campaign.json" \
  "$RRL_ANALYSIS_OUTPUT/analysis_config.json" \
  "$RRL_ANALYSIS_CODE/experiments/audit_bart_fixed_data.py"; do
  [[ -f "$path" ]] || { echo "Missing required file: $path" >&2; exit 2; }
done
command -v sbatch >/dev/null
command -v squeue >/dev/null
command -v sacct >/dev/null
command -v flock >/dev/null
RRL_MAX_SUBMIT_JOBS="${RRL_MAX_SUBMIT_JOBS:-32}"
[[ "$RRL_MAX_SUBMIT_JOBS" =~ ^[1-9][0-9]*$ ]] || { echo 'Invalid submit limit' >&2; exit 2; }
mkdir -p "$(dirname "$RRL_SUBMISSION_DIR")"
if (( resume )); then
  [[ -d "$RRL_SUBMISSION_DIR" ]] || { echo 'No submission directory to resume' >&2; exit 2; }
else
  if ! mkdir "$RRL_SUBMISSION_DIR"; then
    echo "Record exists: inspect $RRL_SUBMISSION_DIR/jobs.tsv and use --resume, not a fresh submission." >&2
    exit 2
  fi
fi
exec 9> "$RRL_SUBMISSION_DIR/.submit.lock"
flock -n 9 || { echo 'Another submitter is active' >&2; exit 2; }
trap 'echo "Submission paused; inspect $RRL_SUBMISSION_DIR/jobs.tsv. Later use --resume; do not delete the ledger." >&2' ERR

# Finished dependency IDs can disappear from scontrol while still being in accounting.
# Remove only dependencies whose recorded terminal state establishes satisfaction.
dependency() {
  local spec="$1" part type id active states state terminal keep result=""
  local parts
  IFS=',' read -r -a parts <<< "$spec"
  for part in "${parts[@]}"; do
    type="${part%%:*}"; id="${part#*:}"
    [[ "$type" == afterok || "$type" == afterany ]] || return 2
    if ! active=$(squeue -h -r -j "$id" -o '%T' 2>/dev/null); then active=""; fi
    keep=1
    if [[ -z "$active" ]]; then
      if ! states=$(sacct -n -X -j "$id" --format=State --parsable2); then
        echo "Cannot verify dependency $id from accounting; no submission made." >&2
        return 2
      fi
      terminal=1
      [[ -n "${states//[[:space:]]/}" ]] || terminal=0
      while IFS='|' read -r state _; do
        state="${state%% *}"; state="${state%+}"
        [[ -n "$state" ]] || continue
        case "$state" in
          COMPLETED|FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED|BOOT_FAIL|DEADLINE|REVOKED) ;;
          *) terminal=0 ;;
        esac
        if [[ "$type" == afterok && "$state" != COMPLETED ]]; then
          echo "Required dependency $id is $state, not COMPLETED. Inspect it before resuming." >&2
          return 2
        fi
      done <<< "$states"
      if (( terminal )); then keep=0; fi
    fi
    if (( keep )); then result="${result:+$result,}$part"; fi
  done
  printf '%s' "$result"
}

submit() {
  local label="$1" raw id existing arg dep tasks=1 count a b
  local args=()
  shift
  if [[ -f "$RRL_SUBMISSION_DIR/jobs.tsv" ]]; then
    existing=$(awk -F '\t' -v key="$label" '$1==key {print $2}' "$RRL_SUBMISSION_DIR/jobs.tsv")
    if [[ -n "$existing" ]]; then
      [[ "$existing" =~ ^[0-9]+$ ]] || { echo "Invalid/duplicate ledger entry: $label" >&2; return 2; }
      echo "Already submitted: $label=$existing" >&2
      printf '%s\n' "$existing"
      return
    fi
  fi
  for arg in "$@"; do
    case "$arg" in
      --dependency=*)
        dep=$(dependency "${arg#--dependency=}") || return 2
        [[ -z "$dep" ]] || args+=("--dependency=$dep") ;;
      --array=*)
        if [[ "${arg#--array=}" =~ ^([0-9]+)-([0-9]+)(%[0-9]+)?$ ]]; then
          a="${BASH_REMATCH[1]}"; b="${BASH_REMATCH[2]}"
          tasks=$((b-a+1)); (( tasks > 0 )) || return 2
        else
          echo "Unsupported array format: $arg" >&2; return 2
        fi
        args+=("$arg") ;;
      *) args+=("$arg") ;;
    esac
  done
  raw=$(squeue -h -r -u "${USER:?}" -o '%i') || return 2
  count=$(printf '%s\n' "$raw" | awk 'NF {n++} END {print n+0}')
  if (( count + tasks > RRL_MAX_SUBMIT_JOBS )); then
    echo "Paused before $label: $count tasks queued/running + $tasks new > limit $RRL_MAX_SUBMIT_JOBS." >&2
    echo 'Let existing tasks finish, then run this launcher with --resume.' >&2
    return 75
  fi
  if ! raw=$(sbatch --parsable "${args[@]}"); then
    echo "Slurm rejected $label; no ID recorded. Inspect the cause, then use --resume." >&2
    return 75
  fi
  id="${raw%%;*}"
  [[ "$id" =~ ^[0-9]+$ ]] || { echo "Unexpected sbatch response: $raw" >&2; return 2; }
  printf '%s\t%s\n' "$label" "$id" >> "$RRL_SUBMISSION_DIR/jobs.tsv"
  printf '%s\n' "$id"
}

prep=$(submit prepare \
  --partition=ialab-low --account=default-account --qos=regular \
  --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=20G --time=00:30:00 \
  --job-name=rrl-expand-prep --output=rrl-expand-prep-%j.out \
  --comment="Experimento rrtl. Supervisor: Guillermo Puebla" <<'SBATCH'
#!/usr/bin/env bash
set -euo pipefail
source "$RRL_ROOT/slurm/common.sh"
python -m experiments.compare_pong_progressive --stage authorize \
  --campaign "$RRL_ORIGINAL_CAMPAIGN" --stop-at 2000000 \
  --reason "Continuar las tres semillas baseline OCAtari hasta 2M"
if (( RRL_OCATARI_ONLY == 0 )); then
  python -m experiments.compare_pong_progressive --stage freeze \
    --protocol configs/pong_online_baseline_visual_2m_extra12.json --campaign "$RRL_VISUAL_EXTRA"
  python -m experiments.compare_pong_progressive --stage authorize \
    --campaign "$RRL_VISUAL_EXTRA" --stop-at 2000000 --reason "Completar quince semillas visuales"
fi
python -m experiments.compare_pong_progressive --stage freeze \
  --protocol configs/pong_online_baseline_ocatari_2m_extra12.json --campaign "$RRL_OCATARI_EXTRA"
python -m experiments.compare_pong_progressive --stage authorize \
  --campaign "$RRL_OCATARI_EXTRA" --stop-at 2000000 --reason "Completar quince semillas OCAtari"
python -m experiments.compare_pong_progressive --stage freeze \
  --protocol configs/breakout_online_baseline_15seeds.json --campaign "$RRL_BREAKOUT_CAMPAIGN"
python -m experiments.compare_pong_progressive --stage authorize \
  --campaign "$RRL_BREAKOUT_CAMPAIGN" --game Breakout --stop-at 100000 \
  --reason "Revisar una semilla a 100k antes de ampliar Breakout"
SBATCH
)

# Two training lanes: Breakout -> extra visual, and existing OCAtari -> extra OCAtari.
bo_train=$(submit breakout_100k_train --dependency="afterok:$prep" \
  --output=rrl-breakout-100k-train-%j.out \
  slurm/09_online_train_cell.sbatch --campaign "$RRL_BREAKOUT_CAMPAIGN" --cell 0)
bo_eval=$(submit breakout_100k_evaluate --dependency="afterok:$prep,afterany:$bo_train" \
  --time=02:00:00 --output=rrl-breakout-100k-eval-%j.out \
  slurm/10_online_evaluate.sbatch --campaign "$RRL_BREAKOUT_CAMPAIGN" --cell 0)
submit breakout_report --dependency="afterok:$prep,afterany:$bo_eval" \
  --output=rrl-breakout-100k-report-%j.out \
  slurm/12_online_report.sbatch --campaign "$RRL_BREAKOUT_CAMPAIGN" >/dev/null

oc_train=$(submit ocatari_original3_train --dependency="afterok:$prep" \
  --array=3-5%1 --time=04:00:00 --output=rrl-oc-original3-train-%A_%a.out \
  slurm/11_online_resume.sbatch --campaign "$RRL_ORIGINAL_CAMPAIGN" \
  --attempt-seconds 14280 --extend-seconds 14400 --extension-id ocatari-2m-01 \
  --reason "Continuar baseline OCAtari desde 500k a 2M")
oc_eval=$(submit ocatari_original3_evaluate --dependency="afterok:$prep,afterany:$oc_train" \
  --array=3-5%1 --time=04:00:00 --output=rrl-oc-original3-eval-%A_%a.out \
  slurm/10_online_evaluate.sbatch --campaign "$RRL_ORIGINAL_CAMPAIGN")
submit original_report --dependency="afterok:$prep,afterany:$oc_eval" \
  --output=rrl-original-final-report-%j.out \
  slurm/12_online_report.sbatch --campaign "$RRL_ORIGINAL_CAMPAIGN" >/dev/null

if (( RRL_OCATARI_ONLY == 0 )); then
  vis_train=$(submit visual_extra12_train --dependency="afterok:$prep,afterany:$bo_train" \
    --array=0-11%1 --time=06:00:00 --output=rrl-vis-extra12-train-%A_%a.out \
    slurm/09_online_train_cell.sbatch --campaign "$RRL_VISUAL_EXTRA" --attempt-seconds 21480)
fi
# Diagnostic branches use the historical analysis code and write only their own campaign.
submit assisted_data --dependency="afterok:$prep" --array="$analysis_array" \
  --partition=ialab-low --account=default-account --qos=regular \
  --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=20G --time=02:00:00 \
  --job-name=rrl-assisted --output=rrl-assisted-%A_%a.out \
  --comment="Experimento rrtl. Supervisor: Guillermo Puebla" <<'SBATCH' >/dev/null
#!/usr/bin/env bash
set -euo pipefail
export RRL_ROOT="$RRL_ANALYSIS_CODE"
source "$RRL_ROOT/slurm/common.sh"
case "$SLURM_ARRAY_TASK_ID" in 0) pipeline=ocatari ;; 1) pipeline=visual ;; *) exit 2 ;; esac
python -m experiments.collect_pong_assisted --analysis "$RRL_ANALYSIS_OUTPUT" --pipeline "$pipeline"
SBATCH

audit_prep=$(submit bart_export --dependency="afterok:$prep" \
  --partition=ialab-low --account=default-account --qos=regular \
  --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=20G --time=01:00:00 \
  --job-name=rrl-bart-export --output=rrl-bart-export-%j.out \
  --comment="Experimento rrtl. Supervisor: Guillermo Puebla" <<'SBATCH'
#!/usr/bin/env bash
set -euo pipefail
export RRL_ROOT="$RRL_ANALYSIS_CODE"
source "$RRL_ROOT/slurm/common.sh"
pipelines=(ocatari)
if (( RRL_OCATARI_ONLY == 0 )); then pipelines+=(visual); fi
for pipeline in "${pipelines[@]}"; do
  python -m experiments.freeze_pong_analysis_data \
    --config "$RRL_ANALYSIS_OUTPUT/analysis_config.json" \
    --output "$RRL_ANALYSIS_OUTPUT" --pipeline "$pipeline"
  python -m experiments.audit_bart_fixed_data \
    --analysis "$RRL_ANALYSIS_OUTPUT" --pipeline "$pipeline" --select-only
done
SBATCH
)
submit bart_base_chains --dependency="afterok:$audit_prep" --array="$analysis_array" \
  --partition=ialab-low --account=default-account --qos=regular \
  --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=20G --time=04:00:00 \
  --job-name=rrl-bart-base --output=rrl-bart-base-%A_%a.out \
  --comment="Experimento rrtl. Supervisor: Guillermo Puebla" <<'SBATCH' >/dev/null
#!/usr/bin/env bash
set -euo pipefail
export RRL_ROOT="$RRL_ANALYSIS_CODE"
source "$RRL_ROOT/slurm/common.sh"
case "$SLURM_ARRAY_TASK_ID" in 0) pipeline=ocatari ;; 1) pipeline=visual ;; *) exit 2 ;; esac
for position in 0 1 2; do
  action=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["actions"][int(sys.argv[2])])' \
    "$RRL_ANALYSIS_OUTPUT/$pipeline/selected_actions.json" "$position")
  for initialization in stumps dispersed; do
    echo "BART base: pipeline=$pipeline action=$action initialization=$initialization"
    python -m experiments.audit_bart_fixed_data \
      --analysis "$RRL_ANALYSIS_OUTPUT" --pipeline "$pipeline" \
      --action "$action" --condition base --initialization "$initialization"
  done
done
SBATCH

# One sequential evaluation job avoids reserving 12 pending array tasks.
if (( RRL_OCATARI_ONLY == 0 )); then
  vis_eval=$(submit visual_extra12_evaluate --dependency="afterok:$prep,afterany:$vis_train" \
    --output=rrl-vis-extra12-eval-%j.out \
    slurm/18_evaluate_cells.sbatch "$RRL_VISUAL_EXTRA" {0..11})
  submit visual_extra12_report --dependency="afterok:$prep,afterany:$vis_eval" \
    --output=rrl-vis-extra12-report-%j.out \
    slurm/12_online_report.sbatch --campaign "$RRL_VISUAL_EXTRA" >/dev/null
fi

extra_train=$(submit ocatari_extra12_train --dependency="afterok:$prep,afterany:$oc_train" \
  --array=0-11%1 --time=06:00:00 --output=rrl-oc-extra12-train-%A_%a.out \
  slurm/09_online_train_cell.sbatch --campaign "$RRL_OCATARI_EXTRA" --attempt-seconds 21480)
extra_eval=$(submit ocatari_extra12_evaluate --dependency="afterok:$prep,afterany:$extra_train" \
  --output=rrl-oc-extra12-eval-%j.out \
  slurm/18_evaluate_cells.sbatch "$RRL_OCATARI_EXTRA" {0..11})
submit ocatari_extra12_report --dependency="afterok:$prep,afterany:$extra_eval" \
  --output=rrl-oc-extra12-report-%j.out \
  slurm/12_online_report.sbatch --campaign "$RRL_OCATARI_EXTRA" >/dev/null

trap - ERR
echo "Recorded jobs (existing IDs are reused; no duplicate submissions):"
cat "$RRL_SUBMISSION_DIR/jobs.tsv"
echo "No reserved test or long BART was submitted. Use --resume after quota pauses."
if (( RRL_OCATARI_ONLY )); then
  echo 'OCAtari-only mode: no new visual workers. Previously submitted visual jobs must be cancelled separately.'
fi
