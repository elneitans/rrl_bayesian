#!/usr/bin/env bash
#SBATCH --comment="Experimento rrtl: validar recuperacion sin cambiar protocolo. Supervisor: Guillermo Puebla"
#SBATCH --partition=ialab-low
#SBATCH --account=default-account
#SBATCH --qos=regular
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --job-name=rrl-recovery-check
set -euo pipefail
export RRL_LOG_DIR="${RRL_IMPROVEMENT_CONTROL:?}/environment/${SLURM_JOB_ID:?}"
source "${RRL_ROOT:?}/slurm/common.sh"
python - <<'PY'
from pathlib import Path
import hashlib, os, tempfile
from experiments.diagnose_rrtl_improvements import load_campaign, restore
from experiments.diagnose_rrtl_revision import fork_identity
from experiments.compare_pong_progressive import default_factory
from experiments.pong_analysis_runtime import checked_read, checked_write, digest

control=Path(os.environ['RRL_IMPROVEMENT_CONTROL']); root=control/'campaign'
m,_=load_campaign(root)
assert digest(m)=='5c68a97ac3ddd52553a705703ef4faf12d014108b41ca16e7b3b84d7eff805b4'
# Check actual allocation and durable write, rather than global NAS free space.
fd,name=tempfile.mkstemp(prefix='.recovery-space-probe-',dir=control)
try:
    with os.fdopen(fd,'wb') as handle:
        for _ in range(64): handle.write(os.urandom(1024*1024))
        handle.flush();os.fsync(handle.fileno())
finally:Path(name).unlink(missing_ok=True)
records=[]
for index,mode,expected in [(8,'stable_td',100000),(9,'double_td',85000)]:
    group=m['groups'][index]
    parent=next(p for p in m['parents'] if (p['criterion'],p['seed'])==(group['criterion'],group['seed']))
    folder=root/'continuations'/group['criterion']/str(group['seed'])/mode
    status=checked_read(folder/'status.json')
    assert status['continuation_decisions']==expected and status['state']=='partial'
    env=default_factory(parent['cell'])()
    try:
        session=restore(folder,status,fork_identity(m,parent,mode),env,m)
        assert session.interactions==500000+expected
        records.append({**group,'mode':mode,'counter':session.interactions,
                        'checkpoint_sha256':status['checkpoint']['sha256']})
    finally:env.close()
checked_write(control/'recovery_preflight.json',{'complete':True,'campaign':digest(m),
    'durable_allocation_probe_bytes':64*1024*1024,'restored':records,
    'original_campaign_untouched':True,'protocol_changed':False})
print({'recovery_preflight_passed':True,'restored':records},flush=True)
PY
