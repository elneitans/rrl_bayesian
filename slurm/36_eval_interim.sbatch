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
#SBATCH --signal=B:USR1@120
#SBATCH --job-name=rrl-interim-eval
set -euo pipefail
control="$HOME/archive/rrl_bayesian_control_20261027"
export RRL_LOG_DIR="$control/environment_interim/${SLURM_JOB_ID:-manual}-${SLURM_STEP_ID:-batch}"
source "${RRL_ROOT:?}/slurm/common.sh"
python - "$@" <<'PY'
import json, os, statistics, sys
from pathlib import Path
from experiments.compare_pong_progressive import evaluate_cell, StopRequest
from bayesian_rrtl.online.protocol import load, select_cell
from bayesian_rrtl.online.multigame_protocol import cell_protocol
from experiments.pong_analysis_runtime import checked_read, checked_write

point=int(sys.argv[1]) if len(sys.argv)>1 else 500000
names=sys.argv[2:] or ['pong_cart','pong_extra_trees','breakout_extra_trees']
allowed={'pong_cart','pong_extra_trees','breakout_extra_trees','pong_bart','breakout_cart','breakout_bart'}
if not names or any(n not in allowed for n in names):raise ValueError('Unknown condition')
base=Path.home()/'archive/rrl_bayesian_long_runs_20261027'
plans=[]
for name in names:
    campaign=base/f'{name}_2m_20261027'
    manifest=load(campaign);cell=select_cell(manifest,0)
    protocol=cell_protocol(manifest,cell)
    if point not in protocol['evaluation']['intermediate']['checkpoints']:
        raise ValueError(f'{point} is not a frozen evaluation milestone for {name}')
    status=checked_read(campaign/cell['id']/'status.json')
    if status['counter']<point:raise ValueError(f'{name} has not reached {point}')
    plans.append((name,campaign,cell['seed']))
results=[]
with StopRequest() as stop:
    for name,campaign,seed in plans:
        blocks=evaluate_cell(campaign,0,point=point,group='intermediate',stop=stop)
        for block in blocks:
            episodes=block['episodes']
            complete=bool(episodes) and all(e['status']=='completed' for e in episodes)
            results.append(dict(condition=name,seed=seed,
                counter=point,checkpoint=block['checkpoint'],state='completed' if complete else 'partial',
                mean_raw_return=statistics.mean(e['raw_return'] for e in episodes) if complete else None,
                episodes=episodes))
        if stop.requested:break
control=Path.home()/'archive/rrl_bayesian_control_20261027'
path=control/f'interim-{point}-{os.environ.get("SLURM_JOB_ID","manual")}-{os.environ.get("SLURM_STEP_ID","batch")}.json'
checked_write(path,dict(results=results,scope='One training seed per condition, two planned intermediate evaluation seeds; not final inference.',reserved_test_executed=False))
print(json.dumps(results,indent=2));print(path)
if any(r['state']!='completed' for r in results):raise SystemExit(3)
PY
