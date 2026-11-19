"""Complete-case analysis for the predeclared stable/guarded RRTL campaign."""
import argparse
import collections
import json
from pathlib import Path
import statistics

from experiments.diagnose_rrtl_improvements import load_campaign
from experiments.pong_analysis_runtime import checked_read,checked_write,digest


def save_csv(path,rows):
    import csv
    if not rows:return
    keys=list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=keys);writer.writeheader();writer.writerows(rows)


def analyze(campaign,output,*,plots=True,loader=load_campaign):
    # Reuse the audited per-transition summarizer, packaged with this source
    # snapshot rather than importing analysis files from the user's workspace.
    from experiments.improvement_trace_summary import trace_summary
    campaign,output=Path(campaign),Path(output);output.mkdir(parents=True,exist_ok=True)
    m,_=loader(campaign);r=checked_read(campaign/'report.json')
    if not r['complete'] or r['missing'] or r['campaign']!=digest(m):raise ValueError('Complete bound report required')
    config=m['config'];rows=[];matched=[];windows=[];events=[]
    for cell in r['continuations']:
        base={k:cell[k] for k in ('criterion','seed','role','mode')}
        folder=campaign/'continuations'/cell['criterion']/str(cell['seed'])/cell['mode'];status=checked_read(folder/'status.json')
        windows.extend({**base,**v} for v in trace_summary(folder,status,config['parent_counter']))
        events.extend({**base,**e} for e in cell['events'])
        for point in range(0,config['continuation_decisions']+1,config['evaluation_interval']):
            data=[e for e in cell['evaluations'] if e['point']==point]
            if len(data)!=config['starts']:raise ValueError('Evaluation coverage mismatch')
            values=[v['raw_return'] for v in data];steps=sum(v['steps'] for v in data)
            rows.append({**base,'point':point,'mean_return':statistics.mean(values),'median_return':statistics.median(values),
                'minimum_return':min(values),'maximum_return':max(values),'starts':len(data),
                'truncated':sum(v['truncated'] for v in data),'no_ball_fraction':sum(v['no_ball_decisions'] for v in data)/steps,
                'fire_fraction':sum(v['actions'].get('FIRE',0) for v in data)/steps,
                **(cell['metrics'] if point==config['continuation_decisions'] else {})})
        baseline=next(c for c in r['continuations'] if c['criterion']==cell['criterion'] and c['seed']==cell['seed'] and c['mode']=='baseline')
        old={e['start']:e['raw_return'] for e in baseline['evaluations'] if e['point']==config['continuation_decisions']}
        new={e['start']:e['raw_return'] for e in cell['evaluations'] if e['point']==config['continuation_decisions']}
        differences=[new[s]-old[s] for s in sorted(old)]
        matched.append({**base,'mean_difference':statistics.mean(differences),'median_difference':statistics.median(differences),
                        'wins':sum(d>0 for d in differences),'ties':sum(d==0 for d in differences),'losses':sum(d<0 for d in differences)})
    expected=len(m['groups'])*len(config['modes'])*config['continuation_decisions']
    if sum(v['decisions'] for v in windows)!=expected:raise ValueError('Incomplete trace coverage')
    if r['exact_baseline_transitions']!=len(m['groups'])*config['continuation_decisions']:raise ValueError('Baseline equivalence incomplete')
    save_csv(output/'evaluations.csv',rows);save_csv(output/'matched_vs_baseline.csv',matched);save_csv(output/'training_windows.csv',windows)
    (output/'events.json').write_text(json.dumps(events,indent=2)+'\n')
    grouped=collections.defaultdict(list)
    for row in matched:grouped[row['criterion'],row['role'],row['mode']].append(row)
    roles=[{'criterion':c,'role':role,'mode':mode,'training_seeds':len(v),
            'mean_seed_difference':statistics.mean(e['mean_difference'] for e in v),
            'minimum_seed_difference':min(e['mean_difference'] for e in v),'maximum_seed_difference':max(e['mean_difference'] for e in v)}
           for (c,role,mode),v in sorted(grouped.items())]
    save_csv(output/'seed_role_summary.csv',roles)
    checked_write(output/'completion.json',{'complete':True,'campaign':digest(m),'report_sha256':digest(r),
        'continuations':len(r['continuations']),'verified_trace_rows':expected,
        'evaluation_rollouts':sum(len(c['evaluations']) for c in r['continuations']),
        'exact_baseline_transitions':r['exact_baseline_transitions'],
        'scope':'Predeclared end-point and complete curves; no reserved test and no automatic winner selection.'})
    labels={'baseline':'Baseline','slow_td':'TD lento','double_td':'Objetivo Double','stable_td':'TD lento + Double',
            'guarded':'Revisión protegida','stable_guarded':'TD estable + revisión','refit_only':'Reestimación sin poda'}
    lines=['# Resultados de mejoras RRTL','',
        'Retornos incrementales desde snapshots; no son puntajes de partidas completas. Comparación exploratoria, sin test reservado.',
        f'Continuaciones completas: {len(r["continuations"])}. Transiciones verificadas: {expected:,}.',
        '', '| Criterio | Seed | Rol | Variante | Media final | Mediana final | Podas | Reemplazos | Refits |',
        '|---|---:|---|---|---:|---:|---:|---:|---:|']
    for row in rows:
        if row['point']==config['continuation_decisions']:
            lines.append(f'|{row["criterion"]}|{row["seed"]}|{row["role"]}|{labels[row["mode"]]}|{row["mean_return"]:.2f}|{row["median_return"]:.2f}|{row["prunes_total"]}|{row["replacements_total"]}|{row["refits_total"]}|')
    lines.extend(['','Los arranques de un mismo seed son dependientes. Los grupos de confirmación no se usaron para desarrollar estas revisiones, pero no son un benchmark ciego. Una media alta aislada no demuestra superioridad general.','',
                  '![Curvas](continuaciones.png)'])
    (output/'resultados.md').write_text('\n'.join(lines)+'\n')
    if plots:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        groups=[(g['criterion'],g['seed']) for g in m['groups']];cols=min(5,len(groups));nr=(len(groups)+cols-1)//cols
        fig,axes=plt.subplots(nr,cols,figsize=(4.4*cols,3.7*nr),squeeze=False)
        colors=('#111827','#2563eb','#0891b2','#059669','#d97706','#dc2626','#9333ea')
        for ax,(criterion,seed) in zip(axes.flat,groups):
            for mode,color in zip(config['modes'],colors):
                values=sorted([v for v in rows if v['criterion']==criterion and v['seed']==seed and v['mode']==mode],key=lambda v:v['point'])
                ax.plot([v['point']/1000 for v in values],[v['mean_return'] for v in values],'-o',ms=3,color=color,label=labels[mode])
            ax.set(title=f'{criterion} · {seed}',xlabel='Miles de decisiones nuevas',ylabel='Retorno medio desde snapshot');ax.grid(alpha=.2)
        for ax in list(axes.flat)[len(groups):]:ax.set_visible(False)
        handles,names=axes[0,0].get_legend_handles_labels();fig.legend(handles,names,loc='upper center',ncol=4,fontsize=9)
        fig.suptitle('RRTL: controles TD y revisiones protegidas',y=1.03);fig.tight_layout(rect=(0,0,1,.82))
        fig.savefig(output/'continuaciones.png',dpi=150,bbox_inches='tight');plt.close(fig)
    print(json.dumps({'complete':True,'continuations':len(r['continuations']),'trace_rows':expected}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--no-plots',action='store_true')
    args=p.parse_args();analyze(args.campaign,args.output,plots=not args.no_plots)
