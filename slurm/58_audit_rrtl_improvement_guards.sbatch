#!/usr/bin/env bash
#SBATCH --comment="Experimento rrtl: auditoria independiente de gates de reparacion. Supervisor: Guillermo Puebla"
#SBATCH --partition=ialab-low
#SBATCH --account=default-account
#SBATCH --qos=regular
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --job-name=rrl-guard-audit
set -euo pipefail
export RRL_LOG_DIR="${RRL_IMPROVEMENT_CONTROL:?}/environment/${SLURM_JOB_ID:?}"
source "${RRL_ROOT:?}/slurm/common.sh"
stage="${1:-smoke}"
[[ "$stage" == smoke || "$stage" == campaign ]] || exit 2
if [[ "$stage" == smoke ]]; then
    PYTHONPATH="$RRL_ROOT/tests:$PYTHONPATH" python -m pytest -q \
        "$RRL_IMPROVEMENT_CONTROL/analysis_code/test_stable_revision.py" -k protected_replacement
fi
exec python "$RRL_IMPROVEMENT_CONTROL/analysis_code/audit_guards.py" \
    --campaign "$RRL_IMPROVEMENT_CONTROL/$stage" \
    --output "$RRL_IMPROVEMENT_CONTROL/${stage}_guard_audit.json"
