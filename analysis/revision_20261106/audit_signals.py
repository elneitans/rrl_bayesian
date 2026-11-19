"""Paired descriptive comparison of Q/target/error moments from committed audits.

Candidate windows are repeated dependent diagnostics, not independent samples.
P-values here are operational scores, not evidence of causal split utility.
"""
import argparse
import collections
import json
from pathlib import Path
import statistics

from build_report import checked, save_csv


def audit(root,output):
    output.mkdir(parents=True,exist_ok=True)
    manifest=checked(root/'manifest.json');threshold=.0001;rows=[];grow=[]
    for status_path in sorted((root/'continuations').glob('*/*/*/status.json')):
        status=checked(status_path);identity=status['identity'];folder=status_path.parent
        for path in sorted((folder/'audits').glob('*.json'),key=lambda p:int(p.stem)):
            if path.stem not in status['checkpoints']:continue
            saved=checked(path)
            if saved['identity']!=identity or saved['counter']!=identity['parent_counter']+int(path.stem):
                raise ValueError('Signal audit frontier/identity mismatch')
            active_nodes={node['node']:node for node in saved['trees']}
            common={'criterion':identity['criterion'],'seed':identity['seed'],'mode':identity['mode'],'point':int(path.stem)}
            for entry in saved['signals']:
                q,t,e=[entry['signals'][signal] for signal in ('updated_q','target','td_error')]
                if q['n']!=t['n'] or q['n']!=e['n'] or q['observed_categories']!=t['observed_categories']:
                    raise ValueError('Different observations for paired signal audit')
                row={**common,'node':entry['node'],'literal':entry['literal'],'depth':entry['depth'],
                    'action':entry['action'],'n':q['n'],'categories':q['observed_categories'],
                    'missing':entry.get('missing',0),'active_at_commit':entry['node'] in active_nodes,
                    'leaf_at_commit':active_nodes.get(entry['node'],{}).get('leaf')}
                for name,value in [('updated_q',q),('target',t),('td_error',e)]:
                    row[name+'_effect']=value['effect_fraction']
                    for test in ('historical','anova'):
                        row[name+'_p_'+test]=value['p_'+test]
                rows.append(row)
            for event in saved['events']:
                if event['kind']=='grow':
                    evidence=event['evidence']
                    grow.append({**common,'counter':event['counter'],'node':event['node'],'action':event['action'],
                        'literal':event['literal'],'active_test':event['test'],'n_at_split':evidence['n'],
                        'effect_at_split':evidence['effect_fraction'],
                        'p_historical':evidence['p_historical'],'p_anova':evidence['p_anova']})
    grouped=collections.defaultdict(list)
    for row in rows:grouped[row['criterion'],row['seed'],row['mode']].append(row)
    summaries=[]
    for (criterion,seed,mode),values in sorted(grouped.items()):
        eligible=[v for v in values if v['n']>128 and v['categories']>=2]
        record={'criterion':criterion,'seed':seed,'mode':mode,'candidate_windows':len(values),
                'paired_eligible_windows':len(eligible),'median_n':statistics.median(v['n'] for v in eligible) if eligible else None}
        for test in ('historical','anova'):
            # Requiring both scores avoids treating undefined dispersion as a failed test.
            joint=[v for v in eligible if v['updated_q_p_'+test] is not None and v['target_p_'+test] is not None]
            record[test+'_q_undefined_target_defined']=sum(v['updated_q_p_'+test] is None and v['target_p_'+test] is not None for v in eligible)
            record[test+'_target_undefined_q_defined']=sum(v['target_p_'+test] is None and v['updated_q_p_'+test] is not None for v in eligible)
            record[test+'_both_undefined']=sum(v['target_p_'+test] is None and v['updated_q_p_'+test] is None for v in eligible)
            qyes=[v['updated_q_p_'+test]<=threshold for v in joint]
            tyes=[v['target_p_'+test]<=threshold for v in joint]
            record[test+'_joint_defined']=len(joint)
            record[test+'_both_below_threshold']=sum(q and t for q,t in zip(qyes,tyes))
            record[test+'_q_only_below_threshold']=sum(q and not t for q,t in zip(qyes,tyes))
            record[test+'_target_only_below_threshold']=sum(t and not q for q,t in zip(qyes,tyes))
        for signal in ('updated_q','target','td_error'):
            record[signal+'_median_effect']=statistics.median(v[signal+'_effect'] for v in eligible) if eligible else None
        summaries.append(record)
    save_csv(output/'paired_signal_windows.csv',rows);save_csv(output/'paired_signal_summary.csv',summaries)
    save_csv(output/'actual_growth_events.csv',grow)
    result={'scope':'Paired repeated candidate-window diagnostics, no independence or causal claim',
        'threshold':threshold,'sample_minimum_strict':128,'summaries':summaries,'growth_events':len(grow),
        'actual_signal_used_to_grow':'updated_q; target and td_error moments were instrumentation only',
        'limitations':['Windows reset every diagnostic commit; they are not lifetime split statistics.',
            'Depth-limited candidates remain in diagnostic counts; passing a score does not imply a split is allowed.',
            'Nodes can grow or be removed during a window; active_at_commit and leaf_at_commit describe its final frontier only.',
            'Internal-node fallback observations are instrumented too; sample eligibility is not structural eligibility.',
            'Different policies visit different states; compare signal pairs within each candidate-window.',
            'Undefined scores are excluded jointly, never imputed as zero or a failed test.',
            'A target-only or Q-only score does not establish which partition improves future return.']}
    (output/'signal_comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'paired_candidate_windows':len(rows),'growth_events':len(grow),'groups':len(summaries)}))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();audit(args.campaign,args.output)
