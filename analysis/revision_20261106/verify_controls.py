"""Compare full normal continuations with original committed real Atari logs.

No model loading, emulator or training. Completed normal cases are verified;
unfinished cases remain missing and prevent a complete result.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

from build_report import checked

FIELDS=('interaction','transition_id','episode_id','step_id','episode_seed','epsilon',
        'action','reward','raw_reward','terminated','truncated','episode_end','raw_episode_return','td')


def original_segments(folder,head):
    segments=[];seen=set();next_start=None
    while head:
        path=folder/head['path']
        if path.resolve().parent != (folder/'logs').resolve() or head['sha256'] in seen:
            raise ValueError('Reference log path/cycle mismatch')
        seen.add(head['sha256']);raw=path.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=head['sha256']:raise ValueError('Reference log checksum mismatch')
        header=json.loads(raw.split(b'\n',1)[0])
        if header['schema']!='pong_log_segment_v2' or header['end']!=head['end']:
            raise ValueError('Reference log header mismatch')
        if next_start is not None and header['end']!=next_start-1:raise ValueError('Reference log gap')
        segments.append((path,header));next_start=header['start'];head=header['previous']
    if next_start!=1:raise ValueError('Reference does not start at one')
    return list(reversed(segments))


def verify(campaign,historical,anova,output):
    manifest=checked(campaign/'manifest.json');config=manifest['config'];results=[];missing=[]
    roots={'historical_f':historical,'anova':anova}
    for group in manifest['groups']:
        criterion,seed=group['criterion'],group['seed']
        folder=campaign/'continuations'/criterion/str(seed)/'normal';status_path=folder/'status.json'
        if not status_path.exists():missing.append(group);continue
        status=checked(status_path)
        if status['state']!='completed':missing.append(group);continue
        parent=next(p for p in manifest['parents'] if p['criterion']==criterion and p['seed']==seed and p['counter']==config['parent_counter'])
        if status['identity']['parent']!=parent['checkpoint'] or status['identity']['mode']!='normal':
            raise ValueError('Normal control parent/type mismatch')
        actual=[];expected_start=config['parent_counter']+1
        for ref in status['traces']:
            path=folder/ref['path'];raw=path.read_bytes()
            if path.resolve().parent!=(folder/'traces').resolve() or hashlib.sha256(raw).hexdigest()!=ref['sha256']:
                raise ValueError('Normal trace checksum/path mismatch')
            if ref['start']!=expected_start:raise ValueError('Normal trace gap')
            chunk=[json.loads(line) for line in gzip.decompress(raw).splitlines()]
            if len(chunk)!=ref['end']-ref['start']+1:raise ValueError('Normal trace length mismatch')
            actual.extend(chunk);expected_start=ref['end']+1
        if len(actual)!=config['continuation_decisions'] or expected_start-1!=status['counter']:
            raise ValueError('Normal trace frontier mismatch')
        source=roots[criterion]/parent['cell']['id'];reference=checked(source/'status.json')
        original_manifest=checked(roots[criterion]/'campaign.json')
        canonical=json.dumps(original_manifest,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
        if hashlib.sha256(canonical).hexdigest()!=parent['campaign_sha256']:
            raise ValueError('Wrong reference campaign')
        compared=0
        for path,header in original_segments(source,reference['log_head']):
            if header['end']<=config['parent_counter']:continue
            if header['start']>status['counter']:break
            with path.open() as handle:
                next(handle)
                for row in map(json.loads,handle):
                    if row['interaction']<=config['parent_counter']:continue
                    if row['interaction']>status['counter']:break
                    observed=actual[compared]
                    for key in FIELDS:
                        if row[key]!=observed[key]:
                            raise ValueError(f'Control differs: {criterion}/{seed}/{row["interaction"]}/{key}')
                    compared+=1
        if compared!=len(actual):raise ValueError('Incomplete reference comparison')
        case={**group,'transitions':compared,'normal_checkpoint':status['checkpoint'],'exact':True}
        results.append(case);print(json.dumps(case),flush=True)
    result={'scope':'Normal control vs pre-existing original continuation; exact observable TD/policy equivalence',
        'complete':not missing and len(results)==len(manifest['groups']),'missing':missing,'cases':results,'fields':FIELDS}
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True);p.add_argument('--historical',type=Path,required=True)
    p.add_argument('--anova',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();verify(a.campaign,a.historical,a.anova,a.output)
