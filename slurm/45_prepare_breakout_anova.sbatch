#!/usr/bin/env bash
#SBATCH --comment="Experimento rrtl. Supervisor: Guillermo Puebla"
#SBATCH --partition=ialab-low
#SBATCH --account=default-account
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --time=12:00:00
#SBATCH --job-name=rrl-anova-prepare
set -euo pipefail
export RRL_LOG_DIR="${RRL_CONTROL:?}/environment/${SLURM_JOB_ID:?}"
source "${RRL_ROOT:?}/slurm/common.sh"
python -m pytest tests/test_baseline_anova.py tests/test_baseline_adapter.py tests/test_baseline_checkpoint.py -m 'not atari and not ocatari' -q
python - <<'PY'
import copy,json,os
from pathlib import Path
from bayesian_rrtl.online.protocol import freeze,authorize,dry_run
from experiments.compare_pong_progressive import train_cell,evaluate_cell,consolidate,measurements

root=Path(os.environ['RRL_ROOT']);control=Path(os.environ['RRL_CONTROL'])
full=json.loads((root/'configs/breakout_anova_10seeds_1m_20261102.json').read_text())
assert full['learners']['corrected_common']['split_test']=='anova'
assert full['games']['Breakout']['training_seeds']==list(range(710601,710611))

def save(p,folder):
    folder.mkdir(parents=True,exist_ok=True);f=folder/'protocol.json'
    if f.exists() and json.loads(f.read_text())!=p:raise ValueError('Refuse changed protocol')
    f.write_text(json.dumps(p,indent=2)+'\n');freeze(p,folder)

for label,maximum in (('smoke',64),('cost',20480)):
    p=copy.deepcopy(full);p['name']=f'breakout_anova_{label}_20261102';p['level']='baseline'
    g=p['games']['Breakout'];g['training_seeds']=[710601]
    p['cells']=[dict(game='Breakout',pipeline='ocatari',learner='corrected_common',seed=710601)]
    g['max_decisions']=maximum;g['milestones']=[maximum];g['auc_interval']=[0,maximum]
    g['evaluation']['intermediate']['checkpoints']=[0,maximum]
    g['evaluation']['final_development']['checkpoints']=[maximum]
    if label=='smoke':
        g['environment']['horizon']=128;g['checkpoint_interval']=16
        for spec in g['evaluation'].values():spec['horizon']=128
        p['learners']['corrected_common']['min_sample_size']=4
    folder=control/label;save(p,folder)
    authorize(folder,maximum,'User requested Breakout ANOVA; integration/cost check before 1M.',game='Breakout')
    state=train_cell(folder,'0',attempt_seconds=1800)
    if state['counter']!=maximum or state['state']!='completed':
        raise SystemExit(f'{label} training partial; inspect/resume preparation')
    evaluate_cell(folder,'0')
    report=consolidate(folder)
    if not report['completed']:raise SystemExit(f'{label} evaluation partial; inspect/resume preparation')
    print(f'{label} validated: {maximum} decisions with development evaluation',flush=True)

cost=measurements(control/'cost')
folder=Path(os.environ['RRL_BREAK_1M_ROOT'])/'corrected_common'
save(full,folder)
(folder/'cost_measurements.json').write_text(json.dumps(cost,indent=2)+'\n')
(folder/'cost_projection.json').write_text(json.dumps(dry_run(full,cost),indent=2)+'\n')
authorize(folder,1000000,'Explicit user request: Breakout baseline with ANOVA; same ten seeds and 1M goal as previous comparison; new Linux cost review; no reserved test.',cost,game='Breakout')
print(f'ANOVA campaign authorized to 1M: {folder}',flush=True)
PY
