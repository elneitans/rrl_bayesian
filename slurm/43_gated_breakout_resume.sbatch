#!/usr/bin/env bash
#SBATCH --comment="Experimento rrtl. Supervisor: Guillermo Puebla"
#SBATCH --partition=ialab-low
#SBATCH --account=default-account
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --time=24:00:00
#SBATCH --signal=B:USR1@180
#SBATCH --job-name=rrl-break-1m-continuation
# Submit one array element with afterany:PREVIOUS_GROUP, never as a daemon.
set -euo pipefail
[[ $# == 2 && "$1" =~ ^(corrected_common|cart|extra_trees|bart)$ && "$2" =~ ^[0-9]+$ ]] || exit 2
method=$1; previous=$2; group=${SLURM_ARRAY_TASK_ID:?}
[[ "$group" =~ ^[0-4]$ ]] || exit 2
known=$(awk -v k="${method}_train" '$1==k {print $2}' "${RRL_CONTROL:?}/breakout_1m_jobs.tsv")
[[ "$known" == "$previous" ]] || { echo 'Previous array does not belong to this condition'; exit 2; }
# Slurm afterany excludes overlap with the original group. The accounting
# database is unreachable from some compute nodes: validate durable campaign
# frontiers here instead of calling sacct on the compute node.
export RRL_GATE_METHOD="$method" RRL_GATE_GROUP="$group"
export RRL_LOG_DIR="$RRL_CONTROL/environment_1m/${SLURM_JOB_ID:?}"
source "${RRL_ROOT:?}/slurm/common.sh"
python - <<'PY'
import os
from pathlib import Path
from bayesian_rrtl.online.protocol import load,select_cell
from experiments.compare_pong_progressive import validate_status
from experiments.pong_analysis_runtime import checked_read
p=Path(os.environ['RRL_BREAK_1M_ROOT'])/os.environ['RRL_GATE_METHOD']
m=load(p);group=int(os.environ['RRL_GATE_GROUP'])
allowed={'insufficient_time_before_fit','attempt_time_exhausted','authorized_milestone_reached'}
for index in (group*2,group*2+1):
    c=select_cell(m,str(index));f=p/c['id']/'status.json'
    if not f.exists():
        if index==group*2:
            raise SystemExit('No first frontier: original group initialization requires review')
        continue  # second cell may not yet have started in the original group
    s=validate_status(checked_read(f),m,c)
    if not s.get('checkpoint') or s['state']=='failed' or s['counter']>1000000:
        raise SystemExit('Invalid/failed frontier; explicit review required')
    if s['counter']<1000000 and (s['state']!='partial' or s['reason'] not in allowed):
        raise SystemExit('Frontier is not a recognized operational pause')
print('Validated group frontiers; resume only unfinished training/evaluation',flush=True)
PY
exec bash "$RRL_CONTROL/slurm/38_breakout_1m_group.sbatch" "$method"
