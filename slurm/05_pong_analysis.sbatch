#!/usr/bin/env bash
#SBATCH --job-name=rrl-bart-analysis
#SBATCH --comment="Experimento rrtl. Supervisor: Guillermo Puebla"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --time=12:00:00
#SBATCH --output=rrl-bart-analysis-%j.out
#SBATCH --partition=ialab-low
#SBATCH --account=default-account
#SBATCH --qos=regular
set -euo pipefail
source "${RRL_ROOT:-${SLURM_SUBMIT_DIR:?}}/slurm/common.sh"
: "${RRL_ANALYSIS_OUTPUT:?Set an independent persistent diagnostic directory}"
RRL_ANALYSIS_CONFIG="${RRL_ANALYSIS_CONFIG:-configs/pong_bart_analysis_v2.json}"
RRL_ANALYSIS_PIPELINE="${RRL_ANALYSIS_PIPELINE:-ocatari}"
RRL_ANALYSIS_SMOKE_ARGS=()
if [[ "${RRL_ANALYSIS_SMOKE:-0}" == 1 ]]; then
  RRL_ANALYSIS_SMOKE_ARGS+=(--smoke)
fi
if [[ "${RRL_ANALYSIS_EXTEND_SECONDS:-0}" != 0 ]]; then
  RRL_ANALYSIS_SMOKE_ARGS+=(--extend-seconds "$RRL_ANALYSIS_EXTEND_SECONDS" --extension-id "${RRL_ANALYSIS_EXTENSION_ID:?Set a unique extension ID}")
fi
RRL_ANALYSIS_DATA_ARGS=()
if [[ -n "${RRL_ANALYSIS_DATASET_FOLDER:-}" ]]; then
  RRL_ANALYSIS_DATA_ARGS+=(--dataset-folder "$RRL_ANALYSIS_DATASET_FOLDER")
fi
RRL_ANALYSIS_COMMON_ARGS=()
if [[ "${RRL_ANALYSIS_COMMON:-0}" == 1 ]]; then
  RRL_ANALYSIS_COMMON_ARGS+=(--common)
fi
RRL_ANALYSIS_AUDIT_ARGS=()
if [[ "${RRL_ANALYSIS_EXTEND_DRAWS:-0}" != 0 ]]; then
  RRL_ANALYSIS_AUDIT_ARGS+=(--extend-draws "$RRL_ANALYSIS_EXTEND_DRAWS")
  if [[ "${RRL_ANALYSIS_EXTEND_SECONDS:-0}" == 0 ]]; then
    RRL_ANALYSIS_AUDIT_ARGS+=(--extension-id "${RRL_ANALYSIS_EXTENSION_ID:?Set a unique extension ID}")
  fi
fi
case "${RRL_ANALYSIS_STAGE:?Set preflight, freeze, heuristics, assisted, audit, regression, or fqi}" in
  preflight)
    python -m experiments.pong_analysis_common --config "$RRL_ANALYSIS_CONFIG" --output "$RRL_ANALYSIS_OUTPUT" ;;
  freeze)
    python -m experiments.freeze_pong_analysis_data --config "$RRL_ANALYSIS_CONFIG" \
      --output "$RRL_ANALYSIS_OUTPUT" --pipeline "$RRL_ANALYSIS_PIPELINE" ;;
  heuristics)
    python -m experiments.evaluate_pong_heuristics --config "$RRL_ANALYSIS_CONFIG" \
      --output "$RRL_ANALYSIS_OUTPUT" --pipeline "$RRL_ANALYSIS_PIPELINE" "${RRL_ANALYSIS_SMOKE_ARGS[@]}" "${RRL_ANALYSIS_COMMON_ARGS[@]}" ;;
  assisted)
    python -m experiments.collect_pong_assisted --analysis "$RRL_ANALYSIS_OUTPUT" \
      --pipeline "$RRL_ANALYSIS_PIPELINE" "${RRL_ANALYSIS_SMOKE_ARGS[@]}" ;;
  collection)
    python -m experiments.collect_pong_dataset --analysis "$RRL_ANALYSIS_OUTPUT" \
      --pipeline "$RRL_ANALYSIS_PIPELINE" "${RRL_ANALYSIS_SMOKE_ARGS[@]}" ;;
  audit)
    python -m experiments.audit_bart_fixed_data --analysis "$RRL_ANALYSIS_OUTPUT" \
      --pipeline "$RRL_ANALYSIS_PIPELINE" --action "${RRL_ANALYSIS_ACTION:?Set predeclared action index}" \
      --condition "${RRL_ANALYSIS_CONDITION:?Set prior sensitivity condition}" \
      --initialization "${RRL_ANALYSIS_INITIALIZATION:-stumps}" "${RRL_ANALYSIS_SMOKE_ARGS[@]}" "${RRL_ANALYSIS_AUDIT_ARGS[@]}" ;;
  regression|fqi)
    python -m experiments.compare_fqi_trees --analysis "$RRL_ANALYSIS_OUTPUT" \
      --pipeline "$RRL_ANALYSIS_PIPELINE" --stage "$RRL_ANALYSIS_STAGE" \
      --dataset "${RRL_ANALYSIS_DATASET:-pilot}" "${RRL_ANALYSIS_SMOKE_ARGS[@]}" "${RRL_ANALYSIS_DATA_ARGS[@]}" ;;
  *) echo 'Invalid RRL_ANALYSIS_STAGE' >&2; exit 2 ;;
esac
