"""Closure verifier for the complete prospective ANOVA confirmation only."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import statistics as stats


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def checked(path, schema='pong_analysis_unit_v2'):
    raw=json.loads(path.read_text())
    if raw['schema']!=schema or raw['sha256']!=digest(raw['payload']):
        raise ValueError('Checksum/schema mismatch: '+str(path))
    return raw['payload']


def verify(campaign,analysis,audit_path,output,parents_verification,integration):
    m=checked(campaign/'manifest.json');report=checked(campaign/'report.json')
    completed=checked(analysis/'completion.json');audit=checked(audit_path)
    parents=checked(parents_verification,'rrtl_confirmation_parents_verification_v1')
    local=checked(analysis/'local_data_verification.json','rrtl_confirmation_local_download_verification_v1')
    smoke=checked(integration/'manifest.json');smoke_report=checked(integration/'report.json')
    if m['schema']!='rrtl_anova_confirmation_v1':raise ValueError('Wrong experiment')
    cfg=m['config'];modes=['baseline','slow_td','refit_only'];seeds=list(range(721101,721106))
    assert cfg['modes']==modes and cfg['seeds']==seeds
    assert cfg['parent_counter']==500000 and cfg['continuation_decisions']==500000
    assert cfg['evaluation_interval']==25000 and cfg['starts']==12 and cfg['evaluation_horizon']==3000
    assert parents['complete'] and parents['verified_transitions']==5000000 and parents['development_rollouts']==55
    assert not parents['reserved_test_run'] and [p['seed'] for p in parents['parents']]==seeds
    assert {p['campaign_sha256'] for p in m['parents']}=={parents['campaign']}
    assert local['complete'] and local['campaign']==digest(m) and local['continuations']==15
    assert local['verified_trace_rows']==7500000 and local['verified_evaluation_rollouts']==3780
    assert local['reproduced_training_windows']==1500 and local['banks_ram_disjoint']
    assert smoke['schema']=='rrtl_confirmation_integration_v1'
    assert smoke['sources']==m['sources'] and smoke['runtime']==m['runtime']
    assert smoke_report['complete'] and not smoke_report['missing'] and smoke_report['campaign']==digest(smoke)
    assert len(smoke_report['continuations'])==3 and smoke_report['exact_baseline_transitions']==1000
    assert set(c['mode'] for c in smoke_report['continuations'])==set(modes)
    assert all(c['state']=='completed' for c in smoke_report['continuations'])
    assert report['complete'] and not report['missing'] and completed['complete'] and audit['complete']
    assert digest(m)==report['campaign']==completed['campaign']==audit['campaign']
    assert completed['report_sha256']==digest(report)
    assert completed['continuations']==15 and completed['verified_trace_rows']==7500000
    assert completed['evaluation_rollouts']==3780 and completed['exact_baseline_transitions']==2500000
    assert report['exact_baseline_transitions']==2500000
    cells={(c['criterion'],c['seed'],c['mode']):c for c in report['continuations']}
    assert len(cells)==15 and set(cells)=={('anova',s,mode) for s in seeds for mode in modes}
    assert all(c['state']=='completed' and c['counter']==1000000 for c in cells.values())
    expected={(c['criterion'],c['seed'],c['mode'],e['counter']-500000,e['kind'])
              for c in cells.values() for e in c['events']
              if e['kind'] in ['prune_review','replacement_review','refit_review'] and e['gate']['accepted']}
    actual={(e['criterion'],e['seed'],e['mode'],e['point'],e['kind']) for e in audit['records']}
    assert expected==actual and len(actual)==audit['accepted_changes']
    assert all(e['loss_reproduced'] for e in audit['records'])
    assert not any(c['metrics'][k] for c in cells.values() for k in ['prunes_total','replacements_total'])
    values={};per_seed=[]
    for (criterion,seed,mode),cell in cells.items():
        identity={(e['point'],e['start']) for e in cell['evaluations']}
        assert len(cell['evaluations'])==len(identity)==252
        assert identity=={(p,s) for p in range(0,500001,25000) for s in range(12)}
        for point in range(0,500001,25000):
            data=[e for e in cell['evaluations'] if e['point']==point]
            assert all(e['horizon']==3000 and e['epsilon']==0 for e in data)
            rs=[e['raw_return'] for e in data]
            per_seed.append({'seed':seed,'mode':mode,'point':point,'mean_return':stats.mean(rs),
                             'median_return':stats.median(rs),'minimum':min(rs),'maximum':max(rs)})
            values[seed,mode,point]=stats.mean(rs)
    assert all(len({values[s,mode,0] for mode in modes})==1 for s in seeds)
    summaries=[]
    for mode in modes:
        delta=[values[s,mode,500000]-values[s,'baseline',500000] for s in seeds]
        rng=random.Random(20261110)
        draws=sorted(stats.mean(rng.choices(delta,k=5)) for _ in range(50000))
        summaries.append({'mode':mode,'training_seeds':5,
            'mean_final_return':stats.mean(values[s,mode,500000] for s in seeds),
            'median_seed_final_mean':stats.median(values[s,mode,500000] for s in seeds),
            'mean_paired_difference':stats.mean(delta),'seed_wins':sum(v>0 for v in delta),
            'seed_ties':sum(v==0 for v in delta),'seed_losses':sum(v<0 for v in delta),
            'bootstrap_descriptive_95_lower':draws[1250],'bootstrap_descriptive_95_upper':draws[48749],
            'seeds_final_below_parent':sum(values[s,mode,500000]<values[s,mode,0] for s in seeds),
            'seeds_final_below_observed_peak':sum(values[s,mode,500000]<max(values[s,mode,p] for p in range(0,500001,25000)) for s in seeds)})
    windows=list(csv.DictReader((analysis/'training_windows.csv').open()))
    assert len(windows)==1500 and sum(int(w['decisions']) for w in windows)==7500000
    output.mkdir(parents=True,exist_ok=True)
    for name,rows in [('method_summary.csv',summaries),('per_seed_curves.csv',per_seed)]:
        with (output/name).open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    result={'complete':True,'campaign':digest(m),'counts':{k:completed[k] for k in
        ['continuations','verified_trace_rows','evaluation_rollouts','exact_baseline_transitions']},
        'parent_transitions':parents['verified_transitions'],'parent_development_rollouts':parents['development_rollouts'],
        'technical_integration_verified':True,'local_trace_and_evaluation_verification':digest(local),
        'parent_verification':digest(parents),
        'independently_reconstructed_changes':len(actual),'summaries':summaries,
        'scope':'Five prospectively fixed new training seeds, shared early snapshots. Post-hoc descriptive bootstrap by training seed; no reserved test or automatic winner selection.'}
    (output/'verified_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['campaign','analysis','audit','output','parents-verification','integration']:
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();verify(a.campaign,a.analysis,a.audit,a.output,a.parents_verification,a.integration)
