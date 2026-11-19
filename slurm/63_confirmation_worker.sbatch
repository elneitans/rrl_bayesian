#!/usr/bin/env bash
#SBATCH --comment="Experimento rrtl: confirmacion prospectiva ANOVA. Supervisor: Guillermo Puebla"
#SBATCH --partition=ialab-low
#SBATCH --account=default-account
#SBATCH --qos=regular
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --signal=B:USR1@180
#SBATCH --job-name=rrl-confirm
set -euo pipefail
export RRL_LOG_DIR="${RRL_CONFIRM_CONTROL:?}/environment/${SLURM_JOB_ID:?}"
export PYTHONDONTWRITEBYTECODE=1
source "${RRL_ROOT:?}/slurm/common.sh"
stage=${1:?stage required}
if [[ "$stage" == parents ]];then
 python - <<'PY'
from pathlib import Path
import fcntl,json,os,time
from bayesian_rrtl.online.protocol import freeze,authorize
from experiments.compare_pong_progressive import train_cell,evaluate_cell,StopRequest
control=Path(os.environ['RRL_CONFIRM_CONTROL']);root=Path(os.environ['RRL_ROOT'])
parents=control/'parents'
# Serialize only idempotent protocol/authorization creation, not training.
with (control/'.parents-setup.lock').open('a+b') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 raw=json.loads((root/'configs/breakout_confirmation_parents_20261109.json').read_text())
 freeze(raw,parents)
 costs=json.loads((control/'cost_measurements.json').read_text())
 authorize(parents,1000000,'Prospective fresh-seed confirmation; measured Linux cost.',costs,game='Breakout')
index=int(os.environ['SLURM_ARRAY_TASK_ID']);assert 0<=index<5
with StopRequest() as stop:
 while not stop.requested:
  remaining=int(os.environ['SLURM_JOB_END_TIME'])-time.time()-300
  if remaining<180:raise SystemExit(3)
  status=train_cell(parents,str(index),attempt_seconds=min(1740,remaining),stop=stop)
  if status['state']=='completed' and status['counter']==1000000:break
  if status.get('reason') not in ('attempt_time_exhausted','interrupted_signal','administrative_pause'):
   raise RuntimeError('Parent stopped; review: '+str(status.get('reason')))
 else:raise SystemExit(3)
 evaluations=evaluate_cell(parents,str(index),stop=stop)
 if not evaluations or any(v['state']!='completed' for v in evaluations):
  raise RuntimeError('Parent development evaluation incomplete; resume explicitly')
print({'parent_complete':index,'counter':status['counter']},flush=True)
PY
 exit $?
fi
config="$RRL_ROOT/configs/breakout_anova_confirmation_20261109.json"
exec python -m experiments.confirm_rrtl_anova --stage "$stage" --output "$RRL_CONFIRM_CONTROL/campaign" \
 --config "$config" --parents "$RRL_CONFIRM_CONTROL/parents" --group "${SLURM_ARRAY_TASK_ID:-0}"
