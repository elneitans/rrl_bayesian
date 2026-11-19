#!/usr/bin/env bash
#SBATCH --comment="Experimento rrtl: recuperar solo unidades incompletas. Supervisor: Guillermo Puebla"
#SBATCH --partition=ialab-low
#SBATCH --account=default-account
#SBATCH --qos=regular
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --signal=B:USR1@120
#SBATCH --job-name=rrl-improve-recover
set -euo pipefail
export RRL_LOG_DIR="${RRL_IMPROVEMENT_CONTROL:?}/environment/${SLURM_JOB_ID:?}"
source "${RRL_ROOT:?}/slurm/common.sh"
python - <<'PY'
from pathlib import Path
import os
from experiments import diagnose_rrtl_improvements as runner
from experiments import diagnose_rrtl_revision as common
from experiments.compare_pong_progressive import StopRequest
from experiments.pong_analysis_runtime import checked_read,digest

root=Path(os.environ['RRL_IMPROVEMENT_CONTROL'])/'campaign'
m,bank=runner.load_campaign(root)
assert digest(m)=='5c68a97ac3ddd52553a705703ef4faf12d014108b41ca16e7b3b84d7eff805b4'
index=int(os.environ['SLURM_ARRAY_TASK_ID']);assert index in (8,9)
group=m['groups'][index]
parent=next(p for p in m['parents'] if (p['criterion'],p['seed'])==(group['criterion'],group['seed']))
with StopRequest() as stop:
    for mode in m['config']['modes']:
        folder=root/'continuations'/group['criterion']/str(group['seed'])/mode
        if (folder/'status.json').exists():
            status=checked_read(folder/'status.json')
            identity=common.fork_identity(m,parent,mode)
            assert status['identity']==identity
            if status['state']=='completed':
                assert status['continuation_decisions']==m['config']['continuation_decisions']
                assert status['counter']==parent['counter']+m['config']['continuation_decisions']
                for point in range(0,m['config']['continuation_decisions']+1,m['config']['evaluation_interval']):
                    for start in bank['starts']:
                        expected={**identity,'checkpoint':status['checkpoints'][str(point)],'point':point,
                            'start_bank':m['start_bank'],'start':start['id'],'epsilon':0.,
                            'horizon':m['config']['evaluation_horizon'],
                            'evaluation_seed':m['config']['evaluation_seed']+start['id']}
                        saved=common.verified_evaluation(folder/'evaluations'/str(point)/str(start['id'])/'state.json',expected)
                        assert saved['status']=='completed'
                print({'preserved_complete_unit':[group['criterion'],group['seed'],mode]},flush=True)
                continue
        if not folder.resolve().is_relative_to(root.resolve()):
            raise ValueError('Recovery may only write to private incomplete units')
        if not runner.continue_one(root,m,bank,parent,mode,stop):
            raise SystemExit(3)
print({'recovered_group':index,'complete':True},flush=True)
PY
