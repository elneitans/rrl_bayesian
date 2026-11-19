#!/usr/bin/env bash
#SBATCH --comment="Experimento de capstone RRTL-ANOVA Breakout. Supervisor: Guillermo Puebla"
#SBATCH --partition=ialab-low
#SBATCH --account=default-account
#SBATCH --qos=regular
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --signal=B:USR1@180
#SBATCH --job-name=rrl-followup
set -euo pipefail
export RRL_LOG_DIR="${RRL_FOLLOWUP:?}/environment/${SLURM_JOB_ID:?}"
export PYTHONDONTWRITEBYTECODE=1
source "${RRL_ROOT:?}/slurm/common.sh"
stage=${1:?stage required}
config="$RRL_ROOT/configs/breakout_rrtl_followup_20261117.json"
if [[ "$stage" == prepare ]]; then
    python -m pytest -q tests/test_stable_revision.py tests/test_tree_revision.py tests/test_baseline_anova.py tests/test_rrtl_followup.py
fi
if [[ "$stage" == factorial || "$stage" == refit || "$stage" == diagnostics ]]; then
    exec python - "$stage" "$RRL_FOLLOWUP/campaign" "$SLURM_ARRAY_TASK_ID" <<'PY'
from pathlib import Path
import sys

from experiments import rrtl_followup as campaign

stage, directory, index = sys.argv[1:]
output = Path(directory)
manifest, bank = campaign.read_campaign(output)
if campaign.checked_read(output / 'preflight.json')['source'] != campaign.digest(manifest):
    raise ValueError('Missing successful preflight for this manifest')
parent = manifest['parents'][int(index)]
arms = {
    'factorial': ('baseline', 'slow_td', 'frozen_td', 'frozen_slow_td'),
    'refit': ('refit_only', 'refit_damped'),
    'diagnostics': ('guarded',),
}
with campaign.StopRequest() as stop:
    for arm in arms[stage]:
        run = campaign.reuse_reference if arm in campaign.REFERENCES else campaign.train
        if not run(output, manifest, bank, parent, arm, stop):
            raise SystemExit(3)
    if stage == 'diagnostics' and not campaign.pruning_pairs(output, manifest, bank, parent, stop):
        raise SystemExit(3)
PY
fi
exec python -m experiments.rrtl_followup "$stage" --output "$RRL_FOLLOWUP/campaign" \
    --config "$config" --parents "$RRL_FOLLOWUP_REFERENCE/parents" \
    --reference "$RRL_FOLLOWUP_REFERENCE/campaign" --index "${SLURM_ARRAY_TASK_ID:-0}"
