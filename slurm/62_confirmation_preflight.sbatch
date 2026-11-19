#!/usr/bin/env bash
#SBATCH --comment="Experimento rrtl: confirmacion ANOVA, validacion y costo. Supervisor: Guillermo Puebla"
#SBATCH --partition=ialab-low
#SBATCH --account=default-account
#SBATCH --qos=regular
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --job-name=rrl-confirm-check
set -euo pipefail
export RRL_LOG_DIR="${RRL_CONFIRM_CONTROL:?}/environment/${SLURM_JOB_ID:?}"
export PYTHONDONTWRITEBYTECODE=1
source "${RRL_ROOT:?}/slurm/common.sh"
python -m pytest -q tests/test_confirmation_design.py tests/test_improvement_runner.py \
    tests/test_revision_runner.py tests/test_stable_revision.py
python - <<'PY'
import copy,json,os
from pathlib import Path
from bayesian_rrtl.online.protocol import freeze,authorize,dry_run
from experiments.compare_pong_progressive import train_cell,evaluate_cell,consolidate,measurements,StopRequest
from experiments.pong_analysis_runtime import checked_write

root=Path(os.environ['RRL_ROOT']);control=Path(os.environ['RRL_CONFIRM_CONTROL'])
protocol=json.loads((root/'configs/breakout_confirmation_parents_20261109.json').read_text())
cost=copy.deepcopy(protocol);cost.update(name='breakout_confirmation_cost_20261109',level='baseline')
g=cost['games']['Breakout'];g.update(training_seeds=[721101],max_decisions=20480,
    milestones=[20480],checkpoint_interval=5120,auc_interval=[0,20480])
cost['cells']=[cost['cells'][0]]
g['evaluation']['intermediate']['checkpoints']=[0,20480]
g['evaluation']['final_development']['checkpoints']=[20480]
folder=control/'cost';freeze(cost,folder)
authorize(folder,20480,'User requested prospective comparison; measure Linux runtime before full campaign.',game='Breakout')
with StopRequest() as stop:
    status=train_cell(folder,'0',attempt_seconds=6000,stop=stop)
    if status['state']!='completed' or status['counter']!=20480:
        raise SystemExit('Cost run partial: inspect exact frontier and resume explicitly.')
    evaluate_cell(folder,'0',stop=stop)
report=consolidate(folder)
if not report['completed']:raise SystemExit('Cost evaluation incomplete')
costs=measurements(folder)
(control/'cost_measurements.json').write_text(json.dumps(costs,indent=2)+'\n')
projection=dry_run(protocol,costs)
assert len(projection['estimates'])==5
(control/'parent_cost_projection.json').write_text(json.dumps(projection,indent=2)+'\n')
checked_write(control/'preflight_complete.json',{'complete':True,'cost_decisions':20480,
    'new_training_seeds':list(range(721101,721106)),'projected_parent_cells':5,
    'long_parent_campaign_authorized':False,'reserved_test_run':False})
print('CONFIRMATION_TESTS_AND_LINUX_COST_PASSED; full training remains separate.',flush=True)
PY
