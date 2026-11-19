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
#SBATCH --job-name=rrl-break-1m-report
set -euo pipefail
[[ $# == 1 ]] || exit 2
export RRL_REPORT_METHOD=$1
export RRL_LOG_DIR="${RRL_CONTROL:?}/environment_1m/${SLURM_JOB_ID:?}"
source "${RRL_ROOT:?}/slurm/common.sh"
python - <<'PY'
import os,json
from pathlib import Path
from experiments.compare_pong_progressive import consolidate
p=Path(os.environ['RRL_BREAK_1M_ROOT'])/os.environ['RRL_REPORT_METHOD']
r=consolidate(p)
summary={'goal_decisions':1000000,'expected_training_seeds':10,'cells':[]}
for c in r['cells']:
    ev=[e for e in c['evaluations'] if e['group']=='final_development' and e['counter']==1000000]
    pending=[e for e in c.get('pending_evaluations',[]) if e['counter']<=1000000]
    summary['cells'].append({'cell':c['cell'],'counter':c.get('counter',0),'evaluations_1m':ev,
        'pending_calendar_through_1m':pending,
        'final_evaluated_1m':c.get('counter',0)>=1000000 and len(ev)==1,
        'complete_1m':c.get('counter',0)>=1000000 and len(ev)==1 and not pending})
summary['complete_1m']=len(summary['cells'])==10 and all(c['complete_1m'] for c in summary['cells'])
report=Path(os.environ.get('RRL_PROGRESS_REPORT',str(p/'progress_1m.json')))
report.write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
raise SystemExit(0 if summary['complete_1m'] else 3)
PY
