#!/usr/bin/env bash
#SBATCH --comment="Experimento rrtl. Supervisor: Guillermo Puebla"
#SBATCH --partition=ialab-low
#SBATCH --account=default-account
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --time=02:00:00
#SBATCH --job-name=rrl-long-prepare
#SBATCH --output=rrl-long-prepare-%j.out
set -euo pipefail
source "${RRL_ROOT:?}/slurm/common.sh"
[[ $# == 2 && "$1" =~ ^(pong|breakout)$ && "$2" =~ ^(cart|extra_trees|bart)$ ]] || exit 2
export RRL_LONG_GAME=$1 RRL_LONG_METHOD=$2
python - <<'PY'
import json, os
from pathlib import Path
from experiments.compare_pong_progressive import measurements, consolidate
from experiments.pong_analysis_runtime import bytes_hash
from bayesian_rrtl.online.protocol import freeze, authorize, dry_run

root=Path(os.environ['RRL_ROOT'])
game=os.environ['RRL_LONG_GAME']; method=os.environ['RRL_LONG_METHOD']
calibration=Path(os.environ.get('RRL_CALIBRATION_RESULTS',str(root/'results')))/f'{game}_{method}_cost_calibration_20261027'
p=json.loads((calibration/'protocol.json').read_text())
if not consolidate(calibration)['completed']:
    raise ValueError('Calibration training/evaluation is not complete; do not authorize long')
cost=measurements(calibration)
if not cost or any(r['decisions']!=20480 for r in cost):
    raise ValueError('Cost calibration incomplete; do not authorize long')
p['name']=f'{game}_{method}_2m_20261027'
p['level']='long'
g=p if game=='pong' else p['games']['Breakout']
g['training_seeds']=list(range(501601,501616)) if game=='pong' else list(range(710601,710611))
g['max_decisions']=2000000
g['milestones']=[100000,500000,2000000]
g['evaluation']['intermediate']['checkpoints']=[0,10000,100000,500000,2000000]
g['evaluation']['final_development']['checkpoints']=[100000,500000,2000000]
# Bounded Slurm attempts remain <=24h; these are cumulative per-cell budgets.
g['budgets']['training_seconds_per_cell']=14*86400 if method=='bart' else 86400
if game=='breakout':
    g['auc_interval']=[0,2000000]
    p['cells']=[dict(game='Breakout',pipeline='ocatari',learner=method,seed=seed) for seed in g['training_seeds']]
if method=='bart':
    artifact=Path(os.environ.get('RRL_BART_REVIEW',str(root/'docs/analisis_bart_20261026.md')))
    p['bart_review']={
      'conclusions':'Exploratory user-requested 2M comparison using the historical 20-tree 50+50 recipe. Fixed-data chains had insufficient mixing; training completion is not evidence of posterior convergence. Keep diagnostics and report limitations.',
      'artifacts':{str(artifact):bytes_hash(artifact.read_bytes())}}
campaign=Path(os.environ.get('RRL_LONG_RESULTS',str(root/'results')))/p['name']
campaign.mkdir(parents=True,exist_ok=True)
protocol=campaign/'protocol.json'
if protocol.exists() and json.loads(protocol.read_text())!=p:raise ValueError('Changed long protocol')
protocol.write_text(json.dumps(p,indent=2)+'\n')
(campaign/'cost_measurements.json').write_text(json.dumps(cost,indent=2)+'\n')
review=dry_run(p,cost)
(campaign/'cost_projection.json').write_text(json.dumps(review,indent=2)+'\n')
freeze(p,campaign)
authorize(campaign,2000000,'User requested all four implemented algorithms through 2M; measured Linux calibration reviewed; reserved test excluded.',cost,game='Breakout' if game=='breakout' else None)
print(campaign,flush=True)
PY
