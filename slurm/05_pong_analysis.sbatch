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
RRL_ANALYSIS_CONFIG="${RRL_ANALYSIS_CONFIG:-configs/pong_bart_analysis_v1.json}"
RRL_ANALYSIS_PIPELINE="${RRL_ANALYSIS_PIPELINE:-ocatari}"
RRL_ANALYSIS_SMOKE_ARGS=()
if [[ "${RRL_ANALYSIS_SMOKE:-0}" == 1 ]]; then
  RRL_ANALYSIS_SMOKE_ARGS+=(--smoke)
fi
case "${RRL_ANALYSIS_STAGE:?Set preflight, freeze, heuristics, assisted, audit, regression, or fqi}" in
  preflight)
    python -m experiments.pong_analysis_common --config "$RRL_ANALYSIS_CONFIG" --output "$RRL_ANALYSIS_OUTPUT" ;;
  freeze)
    python -m experiments.freeze_pong_analysis_data --config "$RRL_ANALYSIS_CONFIG" \
      --output "$RRL_ANALYSIS_OUTPUT" --pipeline "$RRL_ANALYSIS_PIPELINE" ;;
  heuristics)
    python -m experiments.evaluate_pong_heuristics --config "$RRL_ANALYSIS_CONFIG" \
      --output "$RRL_ANALYSIS_OUTPUT" --pipeline "$RRL_ANALYSIS_PIPELINE" "${RRL_ANALYSIS_SMOKE_ARGS[@]}" ;;
  assisted)
    python -m experiments.collect_pong_assisted --analysis "$RRL_ANALYSIS_OUTPUT" ;;
  audit)
    python -m experiments.audit_bart_fixed_data --analysis "$RRL_ANALYSIS_OUTPUT" \
      --pipeline "$RRL_ANALYSIS_PIPELINE" --action "${RRL_ANALYSIS_ACTION:?Set predeclared action index}" \
      --condition "${RRL_ANALYSIS_CONDITION:?Set prior sensitivity condition}" "${RRL_ANALYSIS_SMOKE_ARGS[@]}" ;;
  regression|fqi)
    python -m experiments.compare_fqi_trees --analysis "$RRL_ANALYSIS_OUTPUT" \
      --pipeline "$RRL_ANALYSIS_PIPELINE" --stage "$RRL_ANALYSIS_STAGE" \
      --dataset "${RRL_ANALYSIS_DATASET:-pilot}" "${RRL_ANALYSIS_SMOKE_ARGS[@]}" ;;
  *) echo 'Invalid RRL_ANALYSIS_STAGE' >&2; exit 2 ;;
esac
