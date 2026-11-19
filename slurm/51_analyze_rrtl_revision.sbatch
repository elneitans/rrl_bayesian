#!/usr/bin/env bash
#SBATCH --comment="Experimento rrtl: analisis final y auditoria independiente de poda y senales. Supervisor: Guillermo Puebla"
#SBATCH --partition=ialab-low
#SBATCH --account=default-account
#SBATCH --qos=regular
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --job-name=rrl-revision-analysis
set -euo pipefail
export RRL_LOG_DIR="${RRL_REVISION_CONTROL:?}/environment/${SLURM_JOB_ID:?}"
source "${RRL_ROOT:?}/slurm/common.sh"
RRL_ANALYSIS_CODE="$RRL_REVISION_CONTROL/analysis_code"
RRL_ANALYSIS_OUTPUT="$RRL_REVISION_CONTROL/analysis"
python - <<'PY'
import os
from pathlib import Path
from experiments.pong_analysis_runtime import checked_read
r=checked_read(Path(os.environ['RRL_REVISION_CONTROL'])/'campaign/report.json')
assert r['complete'] and not r['missing']
assert len(r['frozen'])==324 and len(r['continuations'])==30
assert all(len(c['evaluations'])==30 and c['state']=='completed' for c in r['continuations'])
assert r['evaluation_validation_start_ram_disjoint']
PY
python "$RRL_ANALYSIS_CODE/build_report.py" --campaign "$RRL_REVISION_CONTROL/campaign" --output "$RRL_ANALYSIS_OUTPUT"
python "$RRL_ANALYSIS_CODE/audit_signals.py" --campaign "$RRL_REVISION_CONTROL/campaign" --output "$RRL_ANALYSIS_OUTPUT"
python "$RRL_ANALYSIS_CODE/verify_controls.py" --campaign "$RRL_REVISION_CONTROL/campaign" \
    --historical "${RRL_REVISION_HISTORICAL:-$HOME/archive/rrl_bayesian_breakout_1m_20261028/corrected_common}" \
    --anova "${RRL_REVISION_ANOVA:-$HOME/archive/rrl_bayesian_anova_20261102/results/corrected_common}" \
    --output "$RRL_ANALYSIS_OUTPUT/control_equivalence_full.json"
python - <<'PY'
import hashlib,json,os
from pathlib import Path
from experiments.pong_analysis_runtime import checked_read,checked_write,digest
root=Path(os.environ['RRL_REVISION_CONTROL']);out=root/'analysis'
controls=json.loads((out/'control_equivalence_full.json').read_text())
assert controls['complete'] and len(controls['cases'])==6
assert sum(c['transitions'] for c in controls['cases'])==600000
summary=json.loads((out/'summary.json').read_text())
assert summary['complete'] and summary['verified_continuation_transitions']==3000000
report=checked_read(root/'campaign/report.json')
checked_write(out/'completion.json',{'campaign':report['campaign'],'report_sha256':digest(report),
    'analysis_sources':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((root/'analysis_code').glob('*.py'))},
    'frozen_rollouts':324,'continuation_rollouts':900,'continuations':30,'verified_transitions':3000000,
    'exact_control_transitions':600000,'complete':True,
    'scope':'Integrity and analysis pipeline complete; efficacy requires interpretation, not merely passing these checks.'})
print('FINAL_ANALYSIS_AND_INDEPENDENT_CONTROL_AUDIT_COMPLETED',flush=True)
PY
