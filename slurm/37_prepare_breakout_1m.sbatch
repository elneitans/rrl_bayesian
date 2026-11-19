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
#SBATCH --job-name=rrl-break-1m-prepare
set -euo pipefail
export RRL_LOG_DIR="${RRL_CONTROL:?}/environment_1m/${SLURM_JOB_ID:?}"
source "${RRL_ROOT:?}/slurm/common.sh"
python - <<'PY'
import json, os
from pathlib import Path
from experiments.compare_pong_progressive import measurements
from experiments.pong_analysis_runtime import bytes_hash
from bayesian_rrtl.online.protocol import freeze, authorize, dry_run, load
from bayesian_rrtl.online.multigame_protocol import raw_protocol

root=Path(os.environ['RRL_ROOT'])
control=Path(os.environ['RRL_CONTROL'])
output=Path(os.environ['RRL_BREAK_1M_ROOT'])
for method in ('corrected_common','cart','extra_trees','bart'):
    if method=='corrected_common':
        reference=root/'results/breakout_baseline_15_linux_20261026_01'
        p=raw_protocol(load(reference))
    else:
        reference=Path.home()/'archive/rrl_bayesian_calibrations_20261027'/f'breakout_{method}_cost_calibration_20261027'
        p=raw_protocol(load(reference))
    cost=measurements(reference)
    if not cost:
        raise ValueError(f'No measured costs: {method}')
    p['name']=f'breakout_{method}_1m_then_2m_20261028'
    p['level']='long'
    g=p['games']['Breakout']
    g['training_seeds']=list(range(710601,710611))
    g['max_decisions']=2000000
    g['milestones']=[100000,500000,1000000,2000000]
    g['auc_interval']=[0,2000000]
    # Keep both future milestones from the start: authorize only 1M today.
    g['evaluation']['intermediate']['checkpoints']=[0,100000,500000,1000000,2000000]
    g['evaluation']['final_development']['checkpoints']=[1000000,2000000]
    g['budgets']['training_seconds_per_cell']=14*86400 if method=='bart' else 2*86400
    g['budgets']['evaluation_seconds_per_episode']=2400
    p['cells']=[dict(game='Breakout',pipeline='ocatari',learner=method,seed=s) for s in g['training_seeds']]
    if method=='bart':
        artifact=control/'analisis_bart_20261026.md'
        p['bart_review']={'conclusions':'Exploratory historical BART recipe. Prior fixed-data chains mixed insufficiently. User requests ten Breakout seeds at 1M, with later continuation to 2M. Training completion does not establish posterior convergence.',
                          'artifacts':{str(artifact):bytes_hash(artifact.read_bytes())}}
    folder=output/method
    folder.mkdir(parents=True,exist_ok=True)
    path=folder/'protocol.json'
    if path.exists() and json.loads(path.read_text())!=p:
        raise ValueError('Existing protocol differs; refuse overwrite')
    path.write_text(json.dumps(p,indent=2)+'\n')
    (folder/'cost_measurements.json').write_text(json.dumps(cost,indent=2)+'\n')
    (folder/'cost_projection.json').write_text(json.dumps(dry_run(p,cost),indent=2)+'\n')
    freeze(p,folder)
    authorize(folder,1000000,'User prioritizes Breakout, ten seeds per implemented learner at 1M; retain ability to resume to 2M; development only.',cost,game='Breakout')
    print(f'Prepared and authorized to 1M: {folder}',flush=True)
PY
