#!/usr/bin/env bash
#SBATCH --comment="Experimento rrtl: auditoria de cambios inmediatos de politica por poda. Supervisor: Guillermo Puebla"
#SBATCH --partition=ialab-low
#SBATCH --account=default-account
#SBATCH --qos=regular
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --job-name=rrl-policy-audit
set -euo pipefail
export RRL_LOG_DIR="${RRL_REVISION_CONTROL:?}/environment/${SLURM_JOB_ID:?}"
source "${RRL_ROOT:?}/slurm/common.sh"
python "$RRL_REVISION_CONTROL/analysis_code/audit_revision_effects.py" \
    --campaign "$RRL_REVISION_CONTROL/campaign" --output "$RRL_REVISION_CONTROL/analysis"
python - <<'PY'
import hashlib,json,os
from pathlib import Path
from experiments.pong_analysis_runtime import checked_read,checked_write,digest
root=Path(os.environ['RRL_REVISION_CONTROL']);out=root/'analysis'
completion=checked_read(out/'completion.json')
assert completion['complete']
result=json.loads((out/'revision_policy_effects.json').read_text())
assert result['complete']
checked_write(out/'policy_audit_completion.json',{
    'campaign':completion['campaign'],'main_analysis':digest(completion),
    'source_sha256':hashlib.sha256((root/'analysis_code/audit_revision_effects.py').read_bytes()).hexdigest(),
    'result_sha256':hashlib.sha256((out/'revision_policy_effects.json').read_bytes()).hexdigest(),
    'accepted_revisions':result['accepted_revisions'],'complete':True,
    'scope':'Immediate pre/post policy audit; no additional RL training or policy selection.'})
PY
