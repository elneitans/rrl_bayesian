"""Verify final artifacts and summarize matched ablations; never accepts partial data."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import statistics as stats


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def checked(path):
    raw=json.loads(path.read_text())
    assert raw['schema']=='pong_analysis_unit_v2' and raw['sha256']==digest(raw['payload']),str(path)
    return raw['payload']


def save_csv(path,rows):
    with path.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def summarize(campaign,analysis,guard_path,output):
    output.mkdir(parents=True,exist_ok=True)
    m=checked(campaign/'manifest.json');report=checked(campaign/'report.json')
    completion=checked(analysis/'completion.json');guard=checked(guard_path)
    expected='5c68a97ac3ddd52553a705703ef4faf12d014108b41ca16e7b3b84d7eff805b4'
    assert digest(m)==report['campaign']==completion['campaign']==guard['campaign']==expected
    assert report['complete'] and not report['missing'] and completion['complete'] and guard['complete']
    assert completion['report_sha256']==digest(report)
    assert completion['continuations']==70 and completion['verified_trace_rows']==7000000
    assert completion['evaluation_rollouts']==4200 and completion['exact_baseline_transitions']==1000000
    assert report['exact_baseline_transitions']==1000000 and len(report['continuations'])==70
    assert guard['source_sha256']=='9e71f8a0c2e32f75995e51f6e52cd9187900267d60cd0ad3ee2817ae1b68653c'
    cells={(c['criterion'],c['seed'],c['mode']):c for c in report['continuations']}
    assert len(cells)==70
    assert set(cells)=={(g['criterion'],g['seed'],mode) for g in m['groups'] for mode in m['config']['modes']}
    assert all(c['state']=='completed' and c['counter']==600000 and len(c['evaluations'])==60 for c in cells.values())
    count=sum(c['metrics'][k] for c in cells.values() for k in ['prunes_total','replacements_total','refits_total'])
    assert guard['accepted_changes']==len(guard['records'])==count
    assert all(v['loss_reproduced'] for v in guard['records'])
    expected_records={(c['criterion'],c['seed'],c['mode'],e['counter']-500000,e['kind'])
                      for c in cells.values() for e in c['events']
                      if e['kind'] in ['prune_review','replacement_review','refit_review'] and e['gate']['accepted']}
    assert expected_records=={(e['criterion'],e['seed'],e['mode'],e['point'],e['kind']) for e in guard['records']}
    csv_rows=list(csv.DictReader((analysis/'evaluations.csv').open()))
    assert len(csv_rows)==350
    returns={};paired={};curves={}
    for key,c in cells.items():
        for point in [0,25000,50000,75000,100000]:
            values={v['start']:v['raw_return'] for v in c['evaluations'] if v['point']==point}
            assert len(values)==12
            mean=stats.mean(values.values());curves[key+(point,)]=mean
            saved=next(r for r in csv_rows if (r['criterion'],int(r['seed']),r['mode'],int(r['point']))==key+(point,))
            assert float(saved['mean_return'])==mean and float(saved['median_return'])==stats.median(values.values())
            if point==100000:returns[key]=mean
    for key in returns:
        c,seed,mode=key
        paired[key]=returns[key]-returns[c,seed,'baseline']
    table=[];roles=[]
    for criterion in ['historical_f','anova']:
        for mode in m['config']['modes']:
            keys=[(criterion,seed,mode) for seed in m['config']['seeds']]
            values=[returns[k] for k in keys];diffs=[paired[k] for k in keys]
            rng=random.Random(20261109)
            draws=sorted(sum(rng.choice(diffs) for _ in diffs)/len(diffs) for _ in range(50000))
            table.append({'criterion':criterion,'mode':mode,'seeds':5,'mean_return':stats.mean(values),
                'median_seed_mean_return':stats.median(values),'mean_paired_difference':stats.mean(diffs),
                'seed_wins':sum(v>0 for v in diffs),'seed_ties':sum(v==0 for v in diffs),'seed_losses':sum(v<0 for v in diffs),
                'descriptive_bootstrap_95_lower':draws[1250],'descriptive_bootstrap_95_upper':draws[48749],
                'mean_initial_return':stats.mean(curves[k+(0,)] for k in keys),
                'seeds_final_below_initial':sum(returns[k]<curves[k+(0,)] for k in keys),
                'seeds_final_below_observed_peak':sum(returns[k]<max(curves[k+(p,)] for p in [0,25000,50000,75000,100000]) for k in keys)})
            for role in ['development','confirmation']:
                selected=[k for k in keys if m['config']['seed_roles'][str(k[1])]==role]
                ds=[paired[k] for k in selected]
                roles.append({'criterion':criterion,'role':role,'mode':mode,'seeds':len(selected),
                              'mean_paired_difference':stats.mean(ds),'seed_wins':sum(v>0 for v in ds),'seed_losses':sum(v<0 for v in ds)})
    ablations=[]
    for criterion in ['historical_f','anova']:
        for mode,control in [('slow_td','baseline'),('double_td','baseline'),('stable_td','slow_td'),
                             ('guarded','baseline'),('stable_guarded','stable_td'),('refit_only','baseline')]:
            ds=[returns[criterion,s,mode]-returns[criterion,s,control] for s in m['config']['seeds']]
            ablations.append({'criterion':criterion,'mode':mode,'control':control,'seeds':5,
                'mean_difference':stats.mean(ds),'seed_wins':sum(v>0 for v in ds),'seed_ties':sum(v==0 for v in ds),'seed_losses':sum(v<0 for v in ds)})
    protected=[e for c in cells.values() if c['mode'] in ['guarded','stable_guarded'] for e in c['events']
               if e['kind'] in ['prune_review','replacement_review']]
    accepted=[e for e in protected if e['gate']['accepted']]
    summary={'complete':True,'campaign':expected,'counts':{k:completion[k] for k in ['continuations','verified_trace_rows','evaluation_rollouts','exact_baseline_transitions']},
        'methods':table,'roles':roles,'incremental_ablations':ablations,
        'operations':{k:sum(c['metrics'][k] for c in cells.values()) for k in ['grows_total','prunes_total','replacements_total','refits_total','target_syncs']},
        'independent_audit_accepted_changes':count,'protected_proposals':len(protected),'protected_accepted':len(accepted),
        'protected_accepted_with_identical_validation_returns':sum(all(v==0 for v in e['return_guard']['differences']) for e in accepted),
        'confidence_scope':'Post-hoc descriptive bootstrap by five selected training seeds; dependent evaluation starts, no reserved test, no confirmatory significance or automatic winner selection.'}
    save_csv(output/'method_summary.csv',table);save_csv(output/'role_summary.csv',roles);save_csv(output/'incremental_ablations.csv',ablations)
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign',type=Path,required=True);p.add_argument('--analysis',type=Path,required=True)
    p.add_argument('--guard-audit',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();summarize(a.campaign,a.analysis,a.guard_audit,a.output)
