#!/usr/bin/env bash
#SBATCH --comment="Experimento rrtl. Supervisor: Guillermo Puebla"
#SBATCH --partition=ialab-low
#SBATCH --account=default-account
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --job-name=rrl-revision-prepare
#SBATCH --signal=B:USR1@120
set -euo pipefail
export RRL_LOG_DIR="${RRL_REVISION_CONTROL:?}/environment/${SLURM_JOB_ID:?}"
source "${RRL_ROOT:?}/slurm/common.sh"
python -m pytest tests/test_tree_revision.py tests/test_revision_runner.py tests/test_baseline_adapter.py tests/test_baseline_anova.py tests/test_baseline_checkpoint.py -q
python - <<'PY'
import json,os
from pathlib import Path
from experiments.diagnose_rrtl_revision import prepare,frozen_group,continue_group,report
from experiments.compare_pong_progressive import StopRequest
root=Path(os.environ['RRL_ROOT']); control=Path(os.environ['RRL_REVISION_CONTROL'])
historical=Path(os.environ['RRL_REVISION_HISTORICAL']);anova=Path(os.environ['RRL_REVISION_ANOVA'])
config=root/'configs/breakout_revision_diagnostics_20261106.json'
p=json.loads(config.read_text());p.update(seeds=[710603],checkpoints=[500000],continuation_decisions=1000,
    evaluation_interval=250,save_interval=250,starts=2,evaluation_horizon=128,evaluation_epsilons=[0.])
p['revision'].update(min_train=16,min_validation=16,min_validation_blocks=2,min_child=4,
    cooldown=100,buffer_capacity=512,bootstrap_samples=100)
p['probe_blocks']=4;p['probe_horizon']=64
smoke_config=control/'smoke_config.json'
if smoke_config.exists() and json.loads(smoke_config.read_text())!=p:raise ValueError('Changed smoke config')
smoke_config.write_text(json.dumps(p,indent=2)+'\n')
smoke=control/'smoke'
prepare(smoke_config,historical,anova,smoke)
with StopRequest() as stop:
    for group in range(2):
        if not frozen_group(smoke,group,stop):raise SystemExit(3)
        if not continue_group(smoke,group,stop):raise SystemExit(3)
if not report(smoke):raise SystemExit('Real revision integration incomplete')
prepare(config,historical,anova,control/'campaign')
print('REAL_SMOKE_AND_MAIN_PREPARATION_COMPLETED',flush=True)
PY
