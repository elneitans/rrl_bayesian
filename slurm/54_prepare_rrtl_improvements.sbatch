#!/usr/bin/env bash
#SBATCH --comment="Experimento rrtl: validacion de TD estable y revisiones protegidas. Supervisor: Guillermo Puebla"
#SBATCH --partition=ialab-low
#SBATCH --account=default-account
#SBATCH --qos=regular
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --job-name=rrl-improve-prepare
set -euo pipefail
export RRL_LOG_DIR="${RRL_IMPROVEMENT_CONTROL:?}/environment/${SLURM_JOB_ID:?}"
source "${RRL_ROOT:?}/slurm/common.sh"
python -m pytest -q tests/test_stable_revision.py tests/test_improvement_runner.py \
    tests/test_tree_revision.py tests/test_revision_runner.py tests/test_baseline_anova.py \
    tests/test_baseline_adapter.py tests/test_baseline_checkpoint.py
python - <<'PY'
from pathlib import Path
import copy,json,os
from experiments.diagnose_rrtl_improvements import prepare,continue_group,report,load_campaign,return_validator
from experiments.compare_pong_progressive import StopRequest,default_factory
from experiments.diagnose_rrtl_revision import blob_read
from experiments.pong_analysis_runtime import checked_read,checked_write

control=Path(os.environ['RRL_IMPROVEMENT_CONTROL']);root=Path(os.environ['RRL_ROOT'])
main=root/'configs/breakout_rrtl_improvements_20261108.json';cfg=json.loads(main.read_text());smoke=copy.deepcopy(cfg)
smoke.update(seeds=[710603],seed_roles={'710603':'development'},continuation_decisions=1000,
    evaluation_interval=250,save_interval=250,evaluation_horizon=128,starts=2,probe_starts=2,
    return_starts=2,return_horizon=128,probe_blocks=4,probe_horizon=64)
smoke['revision'].update(buffer_capacity=512,min_train=16,min_validation=16,min_validation_blocks=2,
                         min_child=4,cooldown=100,bootstrap_samples=100)
for settings in smoke['variants'].values():settings.update(target_interval=100,return_bootstraps=100)
control.mkdir(parents=True,exist_ok=True);path=control/'smoke_config.json';path.write_text(json.dumps(smoke,indent=2)+'\n')
historical=Path(os.environ['RRL_IMPROVEMENT_HISTORICAL']);anova=Path(os.environ['RRL_IMPROVEMENT_ANOVA'])
prepare(path,historical,anova,control/'smoke')
with StopRequest() as stop:
    for group in range(2):assert continue_group(control/'smoke',group,stop)
assert report(control/'smoke')
m,_=load_campaign(control/'smoke');folder=control/'smoke/continuations/historical_f/710603/baseline'
status=checked_read(folder/'status.json');agent=blob_read(folder,status['checkpoint'])['session']['collector']['agent']
point=smoke['evaluation_interval'];bank=blob_read(control/'smoke',m['return_banks'][str(point)])
env=default_factory(m['parents'][0]['cell'])()
try:
    result=return_validator(env,bank,smoke,point,710603)(agent,copy.deepcopy(agent),{'kind':'identity_validation'})
    assert result['accepted'] and result['mean_difference']==0
    assert [v['trajectory_sha256'] for v in result['baseline']]==[v['trajectory_sha256'] for v in result['candidate']]
finally:env.close()
checked_write(control/'smoke/return_identity_check.json',result)
prepare(main,historical,anova,control/'campaign')
print('IMPROVEMENT_REAL_SMOKE_AND_MAIN_PREPARATION_PASSED',flush=True)
PY
python -m experiments.analyze_rrtl_improvements --campaign "$RRL_IMPROVEMENT_CONTROL/smoke" \
    --output "$RRL_IMPROVEMENT_CONTROL/smoke_analysis"
