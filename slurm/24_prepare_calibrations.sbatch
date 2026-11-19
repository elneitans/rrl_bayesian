#!/usr/bin/env bash
#SBATCH --comment="Experimento rrtl. Supervisor: Guillermo Puebla"
#SBATCH --partition=ialab-low
#SBATCH --account=default-account
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --time=00:30:00
#SBATCH --job-name=rrl-calibration-prepare
#SBATCH --output=rrl-calibration-prepare-%j.out
set -euo pipefail
source "${RRL_ROOT:?}/slurm/common.sh"
python - <<'PY'
import copy, json, os
from pathlib import Path
from importlib.metadata import version
from bayesian_rrtl.online.protocol import freeze, authorize

root=Path(os.environ['RRL_ROOT'])
bart=json.loads((root/'configs/comparison_pong_pilot.json').read_text())['learners']['bart']
fields=('kernel','m','burn_in','draws','k','alpha_tree','beta_tree','max_depth','min_child','nu','noise_quantile','variance_floor')
estimators={
    'cart':dict(max_depth=4,min_child=2,criterion='SSE'),
    'extra_trees':dict(n_estimators=20,max_depth=4,min_samples_leaf=2,n_jobs=1,
                       sklearn_version=version('scikit-learn'),encoding='nominal_missing_v1'),
    'bart':{k:bart[k] for k in fields},
}
for game in ('Pong','Breakout'):
  for method,estimator in estimators.items():
    if game=='Pong':
      p=json.loads((root/'configs/pong_online_baseline_ocatari_2m_extra12.json').read_text())
      p['training_seeds']=[501601]
      g=p
    else:
      p=json.loads((root/'configs/breakout_online_baseline_15seeds.json').read_text())
      g=p['games']['Breakout'];g['training_seeds']=[710601]
      p['cells']=[dict(game=game,pipeline='ocatari',learner=method,seed=710601)]
      g['auc_interval']=[0,20480]
    p['name']=f'{game.lower()}_{method}_cost_calibration_20261027'
    p['level']='baseline'
    p['learners']={method:estimator}
    if game=='Pong':p['pipelines']=['ocatari']
    g['max_decisions']=20480;g['milestones']=[20480]
    g['budgets']['training_seconds_per_cell']=72000
    g['budgets']['evaluation_seconds_per_episode']=2400
    g['evaluation']['intermediate']['checkpoints']=[0,20480]
    g['evaluation']['final_development']['checkpoints']=[20480]
    # Preserve every evaluation seed/horizon for compatible measured cost records.
    path=Path(os.environ.get('RRL_CALIBRATION_RESULTS',str(root/'results')))/p['name']
    path.mkdir(parents=True,exist_ok=True)
    raw=path/'protocol.json'
    if raw.exists() and json.loads(raw.read_text())!=p:raise ValueError('Changed calibration protocol')
    raw.write_text(json.dumps(p,indent=2)+'\n')
    freeze(p,path)
    authorize(path,20480,'Cost calibration before full 2M campaign; BART mixing limitations remain exploratory.',game=game if game=='Breakout' else None)
    print(path,flush=True)
PY
