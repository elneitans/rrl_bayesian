"""Audit available frozen units while workers run; never declares full completion.

Only JSON metadata is required. This does not replace the final Linux audit of
checkpoint blobs, full emulator snapshots, holdouts and continuation traces.
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import statistics

from build_report import checked, save_csv


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def audit(root, output):
    output.mkdir(parents=True,exist_ok=True)
    manifest=checked(root/'manifest.json');campaign=digest(manifest)
    config=manifest['config'];starts=checked(root/'starts.json')
    assert len(starts)==config['starts']
    assert len({s['ram_sha256'] for s in starts})==len(starts)
    assert len({s['encoded_hash'] for s in starts})>=min(4,len(starts))
    rows=[];missing=[]
    for parent in manifest['parents']:
        for epsilon in config['evaluation_epsilons']:
            for start in starts:
                folder=root/'frozen'/parent['criterion']/str(parent['seed'])/str(parent['counter'])/str(epsilon)/str(start['id'])
                if not (folder/'state.json').exists():
                    missing.append(str(folder.relative_to(root)));continue
                state=checked(folder/'state.json');identity=checked(folder/'identity.json')
                expected={'campaign':campaign,'parent':parent['checkpoint'],'start_bank':manifest['start_bank'],
                    'start':start['id'],'epsilon':epsilon,'evaluation_seed':config['evaluation_seed']+start['id'],
                    'horizon':config['evaluation_horizon'],'policy':'frozen_epsilon_greedy_uniform_ties'}
                assert identity==expected and state['identity_sha256']==digest(expected),folder
                if state['status']!='completed':missing.append(str(folder.relative_to(root)));continue
                result=state['result']
                assert result['physical_hash']==start['physical_hash'] and result['start']==start['id']
                assert result['seed']==expected['evaluation_seed'] and result['epsilon']==epsilon
                assert result['horizon']==config['evaluation_horizon'] and result['return_scope']=='snapshot_continuation'
                assert 0<result['steps']<=result['horizon'] and sum(result['actions'].values())==result['steps']
                assert 0<=result['no_ball_decisions']<=result['steps']
                assert result['terminated'] != result['truncated']
                rows.append({'criterion':parent['criterion'],'seed':parent['seed'],'counter':parent['counter'],
                    'epsilon':epsilon,'start':start['id'],'return':result['raw_return'],'steps':result['steps'],
                    'truncated':result['truncated'],'no_ball':result['no_ball_decisions'],'lives_lost':result['life_losses'],
                    'fire':result['actions']['FIRE'],'trajectory':result['trajectory_sha256']})
    grouped=collections.defaultdict(list)
    for row in rows:grouped[row['criterion'],row['seed'],row['counter'],row['epsilon']].append(row)
    summaries=[]
    for (criterion,seed,counter,epsilon),values in sorted(grouped.items()):
        summaries.append({'criterion':criterion,'seed':seed,'counter':counter,'epsilon':epsilon,
            'available_starts':len(values),'expected_starts':config['starts'],
            'mean_snapshot_return':statistics.mean(r['return'] for r in values),
            'min_snapshot_return':min(r['return'] for r in values),'max_snapshot_return':max(r['return'] for r in values),
            'truncated':sum(r['truncated'] for r in values),'no_ball_fraction':sum(r['no_ball'] for r in values)/sum(r['steps'] for r in values),
            'fire_fraction':sum(r['fire'] for r in values)/sum(r['steps'] for r in values),
            'unique_trajectories':len({r['trajectory'] for r in values})})
    save_csv(output/'frozen_available_episodes.csv',rows);save_csv(output/'frozen_available_summary.csv',summaries)
    result={'scope':'Partial observational JSON audit; not final campaign completion or checkpoint verification',
        'campaign':campaign,'available':len(rows),'expected':len(manifest['parents'])*len(config['evaluation_epsilons'])*config['starts'],
        'missing_units':missing,'summary':summaries}
    (output/'frozen_partial.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('available','expected','scope')}))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();audit(args.campaign,args.output)
