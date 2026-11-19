#!/usr/bin/env bash
#SBATCH --comment="Experimento rrtl. Supervisor: Guillermo Puebla"
#SBATCH --partition=ialab-low
#SBATCH --account=default-account
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --job-name=rrl-progress-snapshot
#SBATCH --output=rrl-progress-snapshot-%j.out
set -euo pipefail
source "${RRL_ROOT:?}/slurm/common.sh"
python - <<'PY'
import hashlib, json, os, time
from pathlib import Path

root=Path(os.environ['RRL_ROOT'])
existing=Path(os.environ.get('RRL_EXISTING_RESULTS',str(root/'results')))
long=Path(os.environ.get('RRL_LONG_RESULTS',str(Path.home()/'archive/rrl_bayesian_long_runs_20261027')))
control=Path(os.environ.get('RRL_PROGRESS_CONTROL',str(Path.home()/'archive/rrl_bayesian_control_20261027')))
def read(path):
    value=json.loads(path.read_text())
    payload=value['payload']
    sha=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    if value['sha256']!=sha:raise ValueError(f'JSON checksum mismatch: {path}')
    return payload

rows=[]
for game,seeds in [('Pong',range(501601,501616)),('Breakout',range(710601,710611))]:
  for method in ('corrected_common','cart','extra_trees','bart'):
    for seed in seeds:
      if method=='corrected_common':
        name=('pong_baseline_progressive_linux_20261014_01' if seed<=501603 else 'pong_ocatari_2m_extra12_20261026_01') if game=='Pong' else 'breakout_baseline_15_linux_20261026_01'
        campaign=existing/name
      else:
        campaign=long/f'{game.lower()}_{method}_2m_20261027'
      prefix=Path('ocatari') if game=='Pong' else Path(game)/'ocatari'
      folder=campaign/prefix/method/str(seed)
      row=dict(game=game,method=method,seed=seed,campaign=str(campaign),counter=0,state='not_started',evaluation_2m_completed=0,errors=[])
      status=None
      try:
        if (folder/'status.json').exists():
          status=read(folder/'status.json')
          row.update(counter=status['counter'],state=status['state'],reason=status.get('reason'),completed_training_games=status.get('metrics',{}).get('completed_games'))
        if (campaign/'campaign.json').exists():
          manifest=read(campaign/'campaign.json')
          for cell in manifest['cells']:
            if cell['seed']==seed and cell['learner']==method and cell.get('game','Pong')==game and cell['pipeline']=='ocatari':
              protocol=manifest['protocol']
              if manifest['schema']=='atari_comparison_v3':protocol=protocol['games'][game]
              row['evaluation_2m_expected']=len(protocol['evaluation']['final_development']['seeds'])
              if status:
                campaign_sha=hashlib.sha256(json.dumps(manifest,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
                if status['binding']['campaign']!=campaign_sha or status['binding']['cell']!=cell['hash']:
                  raise ValueError('Status campaign/cell binding mismatch')
              break
        for path in (folder/'evaluations/final_development/2000000').glob('*/state.json'):
          episode=read(path)
          if episode['status']=='completed':row['evaluation_2m_completed']+=1
      except Exception as exc:
        row['errors'].append(f'{type(exc).__name__}: {exc}')
      rows.append(row)
summary=[]
for game in ('Pong','Breakout'):
  for method in ('corrected_common','cart','extra_trees','bart'):
    subset=[r for r in rows if r['game']==game and r['method']==method]
    summary.append(dict(game=game,method=method,expected_models=len(subset),
      status_claims_2m=sum(r['counter']==2000000 and r['state']=='completed' and not r['errors'] for r in subset),
      published_decisions=sum(r['counter'] for r in subset),
      completed_final_episodes=sum(r['evaluation_2m_completed'] for r in subset),errors=sum(bool(r['errors']) for r in subset)))
result=dict(unix_time=time.time(),summary=summary,cells=rows,
  limitation='Status JSON hashes/bindings only: does not verify checkpoint blobs, transition chains, source/runtime identities or episode-to-checkpoint bindings. Full completion still requires the existing loaders and reports. Snapshot is not a completion certificate.')
control.mkdir(parents=True,exist_ok=True)
path=control/f'progress-{os.environ.get("SLURM_JOB_ID","manual")}.json'
temp=path.with_suffix('.tmp');temp.write_text(json.dumps(result,indent=2)+'\n');temp.replace(path)
print(json.dumps(summary,indent=2));print(path)
PY
