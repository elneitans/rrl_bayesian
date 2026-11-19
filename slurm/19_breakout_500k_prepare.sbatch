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
#SBATCH --job-name=rrl-breakout-prepare
#SBATCH --output=rrl-breakout-prepare-%j.out
set -euo pipefail
source "${RRL_ROOT:?}/slurm/common.sh"
python - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ['RRL_ROOT'])
campaign = Path(os.environ['RRL_BREAKOUT_CART'])
p = json.loads((root/'configs/breakout_online_baseline_15seeds.json').read_text())
p['name'] = 'breakout_cart_10seeds_500k_v1'
p['learners'] = {'cart': {'max_depth': 4, 'min_child': 2, 'criterion': 'SSE'}}
g = p['games']['Breakout']
g['training_seeds'] = list(range(710601, 710611))
g['max_decisions'] = 500000
g['milestones'] = [100000, 500000]
g['auc_interval'] = [0, 500000]
for evaluation in g['evaluation'].values():
    evaluation['checkpoints'] = [n for n in evaluation['checkpoints'] if n <= 500000]
p['cells'] = [dict(game='Breakout', pipeline='ocatari', learner='cart', seed=seed)
              for seed in g['training_seeds']]
campaign.mkdir(parents=True, exist_ok=True)
path = campaign/'protocol.json'
if path.exists():
    if json.loads(path.read_text()) != p:
        raise ValueError('Existing CART protocol differs; do not overwrite')
else:
    path.write_text(json.dumps(p, indent=2)+'\n')
PY
python -m experiments.compare_pong_progressive --stage freeze \
  --protocol "$RRL_BREAKOUT_CART/protocol.json" --campaign "$RRL_BREAKOUT_CART"
for campaign in "$RRL_BREAKOUT_BASE" "$RRL_BREAKOUT_CART"; do
  python -m experiments.compare_pong_progressive --stage authorize \
    --campaign "$campaign" --game Breakout --stop-at 500000 \
    --reason 'Comparar baseline y CART con las semillas 710601-710610 a 500k; sin test reservado.'
done
