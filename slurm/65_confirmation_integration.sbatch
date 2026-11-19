#!/usr/bin/env bash
#SBATCH --comment="Experimento rrtl: integracion real tres variantes. Supervisor: Guillermo Puebla"
#SBATCH --partition=ialab-low
#SBATCH --account=default-account
#SBATCH --qos=regular
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --job-name=rrl-confirm-smoke
set -euo pipefail
export RRL_LOG_DIR="${RRL_CONFIRM_CONTROL:?}/environment/${SLURM_JOB_ID:?}"
source "${RRL_ROOT:?}/slurm/common.sh"
python - <<'PY'
from pathlib import Path
import copy,json,os
from bayesian_rrtl.online.protocol import runtime,source_hashes
from experiments import diagnose_rrtl_revision as common
from experiments import diagnose_rrtl_improvements as improved
from experiments.confirm_rrtl_anova import MODES
from experiments.compare_pong_progressive import StopRequest
from experiments.pong_analysis_runtime import checked_read,digest,checked_write

control=Path(os.environ['RRL_CONFIRM_CONTROL']);root=Path(os.environ['RRL_ROOT'])
cfg=json.loads((root/'configs/breakout_anova_confirmation_20261109.json').read_text())
cfg=copy.deepcopy(cfg);cfg.update(seeds=[721101],seed_roles={'721101':'confirmation'},
    continuation_decisions=1000,evaluation_interval=250,save_interval=250,starts=2,
    probe_starts=2,return_starts=2,evaluation_horizon=128,return_horizon=128,probe_blocks=4,probe_horizon=64)
cfg['revision'].update(buffer_capacity=512,min_train=16,min_validation=16,min_validation_blocks=2,
                       min_child=4,cooldown=100,bootstrap_samples=100)
path=control/'integration_config.json';path.write_text(json.dumps(cfg,indent=2)+'\n')
out=control/'integration'
def loader(output):
    m=checked_read(Path(output)/'manifest.json')
    assert m['schema']=='rrtl_confirmation_integration_v1' and m['config']==cfg
    assert m['sources']==source_hashes() and m['runtime']==runtime('corrected_common')
    return m,common.blob_read(output,m['start_bank'])
common.prepare(path,control/'parents',control/'parents',out,expected_modes=MODES,
    manifest_schema='rrtl_confirmation_integration_v1',existing_loader=loader,
    extend_manifest=improved.extend_manifest,criteria=('anova',))
m,bank=loader(out);parent=m['parents'][0]
with StopRequest() as stop:
    for mode in MODES:
        assert improved.continue_one(out,m,bank,parent,mode,stop)
assert improved.report(out,loader=loader)
report=checked_read(out/'report.json')
assert len(report['continuations'])==3 and report['exact_baseline_transitions']==1000
checked_write(control/'integration_complete.json',{'complete':True,'campaign':digest(m),
    'continuations':3,'exact_baseline_transitions':1000,'scope':'Reduced technical integration only; not efficacy.'})
PY
