#!/usr/bin/env bash
#SBATCH --comment="Experimento rrtl: comprobar control incremental frente a logs originales. Supervisor: Guillermo Puebla"
#SBATCH --partition=ialab-low
#SBATCH --account=default-account
#SBATCH --qos=regular
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --job-name=rrl-revision-equivalence
set -euo pipefail
source "${RRL_ROOT:?}/slurm/common.sh"
python - <<'PY'
import gzip,json,os
from pathlib import Path
from bayesian_rrtl.online.storage import stream_rows
from experiments.pong_analysis_runtime import checked_read,checked_write,bytes_hash

root=Path(os.environ['RRL_REVISION_CONTROL'])/'smoke'
manifest=checked_read(root/'manifest.json')
assert checked_read(root/'report.json')['complete']
fields=('interaction','transition_id','episode_id','step_id','episode_seed','epsilon',
    'action','reward','raw_reward','terminated','truncated','episode_end','raw_episode_return','td')
results=[]
for parent in manifest['parents']:
    folder=root/'continuations'/parent['criterion']/str(parent['seed'])/'normal'
    status=checked_read(folder/'status.json');actual=[]
    for ref in status['traces']:
        raw=(folder/ref['path']).read_bytes();assert bytes_hash(raw)==ref['sha256']
        actual.extend(json.loads(line) for line in gzip.decompress(raw).splitlines())
    assert len(actual)==manifest['config']['continuation_decisions']
    source=Path(parent['folder']);frontier=checked_read(source/'status.json')
    compared=0
    for row in stream_rows(source,frontier['log_head']):
        if row['interaction']<=parent['counter']:continue
        counterpart=actual[compared]
        for field in fields:
            if row[field]!=counterpart[field]:
                raise AssertionError((parent['criterion'],parent['seed'],row['interaction'],field,row[field],counterpart[field]))
        compared+=1
        if compared==len(actual):break
    assert compared==len(actual)
    result={'criterion':parent['criterion'],'seed':parent['seed'],'compared_transitions':compared,
        'fields':fields,'exact':True}
    results.append(result);print(json.dumps(result),flush=True)
checked_write(root/'control_equivalence.json',{'exact':True,'cases':results,
    'scope':'Real normal control matches the pre-existing original continuation; excludes instrumentation/timing fields.'})
PY
