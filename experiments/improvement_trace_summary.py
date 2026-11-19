"""Audited trace summarizer shared with the completed revision campaign."""
import collections
import gzip
import hashlib
import json
import statistics

def trace_summary(folder, status, parent):
    summaries=[];bins=collections.defaultdict(lambda: {'n':0,'raw':0.,'td_abs':0.,'margin':0.,'ended':[],'all_ended':0,
        'fallback':0,'low_margin':0,'td_action_changes':0,'moves':collections.Counter(),'first':None,'last':None})
    next_counter=parent+1
    # An episode already in progress at the parent checkpoint contains rewards
    # before the fork. Exclude it from per-episode continuation averages.
    first_episode=None
    first_episode_complete=False
    for ref in status['traces']:
        path=folder/ref['path'];raw=path.read_bytes()
        if path.resolve().parent != (folder/'traces').resolve():
            raise ValueError('Trace path outside its experiment directory')
        if hashlib.sha256(raw).hexdigest()!=ref['sha256'] or ref['start']!=next_counter:
            raise ValueError('Trace identity/continuity mismatch')
        rows=[json.loads(line) for line in gzip.decompress(raw).splitlines()]
        if len(rows)!=ref['end']-ref['start']+1:raise ValueError('Trace size mismatch')
        for expected,row in enumerate(rows,ref['start']):
            if row['interaction']!=expected or row['transition_id']!=expected-1:
                raise ValueError('Trace transition counter mismatch')
            bucket=bins[(expected-parent-1)//5000]
            if first_episode is None:
                first_episode=row['episode_id'];first_episode_complete=row['step_id']==0
            bucket['n']+=1;bucket['raw']+=row['raw_reward']
            td=row['td'];bucket['td_abs']+=abs(td['target']-td['old_q'])
            detail=row['diagnostic'];bucket['margin']+=detail['greedy_margin']
            bucket['fallback']+=int(detail['internal_fallback'])
            bucket['low_margin']+=int(detail['greedy_margin']<.01)
            qb,qa=detail['q_before'],detail['q_after']
            best_before={i for i,x in enumerate(qb) if x==max(qb)}
            best_after={i for i,x in enumerate(qa) if x==max(qa)}
            bucket['td_action_changes']+=int(not best_before & best_after)
            bucket['moves'][row['action']]+=1
            bucket['first']=row['raw_episode_return'] if bucket['first'] is None else bucket['first']
            bucket['last']=row['raw_episode_return']
            bucket['all_ended']+=int(row['episode_end'])
            if row['episode_end'] and (row['episode_id']!=first_episode or first_episode_complete):
                bucket['ended'].append(row['raw_episode_return'])
        next_counter=ref['end']+1
    if next_counter!=status['counter']+1:raise ValueError('Trace frontier incomplete')
    for b,v in sorted(bins.items()):
        summaries.append({'point_start':b*5000,'point_end':min((b+1)*5000,status['counter']-parent),'decisions':v['n'],
            'raw_reward':v['raw'],'raw_per_1000_decisions':1000*v['raw']/v['n'],
            'ended_episodes':v['all_ended'],'complete_new_episodes':len(v['ended']),
            'mean_ended_episode_return':statistics.mean(v['ended']) if v['ended'] else None,
            'mean_abs_td_error':v['td_abs']/v['n'],'mean_greedy_margin':v['margin']/v['n'],
            'low_margin_fraction':v['low_margin']/v['n'],'fallback_fraction':v['fallback']/v['n'],
            'disjoint_argmax_before_after_td_fraction':v['td_action_changes']/v['n'],
            **{f'action_{i}_fraction':v['moves'][i]/v['n'] for i in range(4)}})
    return summaries


