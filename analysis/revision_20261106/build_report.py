"""Render audited revision observations; diagnostics are not complete-game scores.

Usage: python analysis/revision_20261106/build_report.py --campaign PATH --output PATH
Reads checked report/manifest plus committed compressed traces; no model loading.
"""
import argparse
import collections
import csv
import gzip
import hashlib
import json
from pathlib import Path
import statistics


def checked(path):
    doc = json.loads(Path(path).read_text())
    if doc.get('schema') != 'pong_analysis_unit_v2':
        raise ValueError(f'Unexpected metadata schema: {path}')
    raw = json.dumps(doc['payload'],sort_keys=True,separators=(',', ':'),allow_nan=False).encode()
    if hashlib.sha256(raw).hexdigest() != doc['sha256']:
        raise ValueError(f'Checksum mismatch: {path}')
    return doc['payload']


def save_csv(path, rows):
    if not rows: return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def mean(rows, key):
    return statistics.mean(row[key] for row in rows) if rows else None


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


def build(campaign, output, *, make_plots=True):
    output.mkdir(parents=True,exist_ok=True)
    report=checked(campaign/'report.json');manifest=checked(campaign/'manifest.json')
    manifest_sha=hashlib.sha256(json.dumps(manifest,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    if report['campaign']!=manifest_sha:raise ValueError('Report campaign binding mismatch')
    frozen=report['frozen'];frozen_summary=[]
    groups=collections.defaultdict(list)
    for row in frozen:groups[row['criterion'],row['seed'],row['counter'],row['epsilon']].append(row)
    for (criterion,seed,counter,epsilon),rows in sorted(groups.items()):
        frozen_summary.append({'criterion':criterion,'seed':seed,'counter':counter,'epsilon':epsilon,
            'mean_snapshot_return':mean(rows,'raw_return'),'median_snapshot_return':statistics.median(r['raw_return'] for r in rows),
            'starts':len(rows),'truncated':sum(r['truncated'] for r in rows),
            'mean_steps':mean(rows,'steps'),'no_ball_fraction':sum(r['no_ball_decisions'] for r in rows)/sum(r['steps'] for r in rows),
            'unique_trajectories':len(set(r['trajectory_sha256'] for r in rows))})
    continue_rows=[];events=[];signals=[];train=[];matched=[]
    per_group={}
    for cell in report['continuations']:
        criterion,seed,mode=cell['criterion'],cell['seed'],cell['mode']
        folder=campaign/'continuations'/criterion/str(seed)/mode
        status=checked(folder/'status.json')
        for row in trace_summary(folder,status,report['config']['parent_counter']):
            train.append({'criterion':criterion,'seed':seed,'mode':mode,**row})
        for row in cell['events']:events.append({'criterion':criterion,'seed':seed,'mode':mode,**row})
        for row in cell['signal_audit']:signals.append({'criterion':criterion,'seed':seed,'mode':mode,**row})
        bypoint=collections.defaultdict(list)
        for e in cell['evaluations']:bypoint[e['point']].append(e)
        for point,rows in sorted(bypoint.items()):
            continue_rows.append({'criterion':criterion,'seed':seed,'mode':mode,'point':point,
                'mean_snapshot_return':mean(rows,'raw_return'),'starts':len(rows),'truncated':sum(r['truncated'] for r in rows),
                'mean_steps':mean(rows,'steps'),**(cell['metrics'] if point==report['config']['continuation_decisions'] else {})})
        per_group[criterion,seed,mode]=bypoint
    for criterion,seed in sorted({(c['criterion'],c['seed']) for c in report['continuations']}):
        normal=per_group.get((criterion,seed,'normal'),{})
        for mode in report['config']['modes']:
            rows=per_group.get((criterion,seed,mode),{})
            point=report['config']['continuation_decisions']
            baseline={r['start']:r['raw_return'] for r in normal.get(point,[])}
            vals=[r['raw_return']-baseline[r['start']] for r in rows.get(point,[]) if r['start'] in baseline]
            matched.append({'criterion':criterion,'seed':seed,'mode':mode,'point':point,
                'paired_starts':len(vals),'mean_difference_vs_normal':statistics.mean(vals) if vals else None})
    save_csv(output/'frozen_summary.csv',frozen_summary);save_csv(output/'continuation_summary.csv',continue_rows)
    save_csv(output/'matched_vs_normal.csv',matched);save_csv(output/'training_windows.csv',train)
    # Nested values remain readable JSON rather than escaped CSV objects.
    (output/'events.json').write_text(json.dumps(events,indent=2))
    save_csv(output/'signal_windows.csv',signals)
    summary={'complete':report['complete'],'missing':report['missing'],'start_diversity':report['start_diversity'],
        'evaluation_validation_start_ram_disjoint':report.get('evaluation_validation_start_ram_disjoint',False),
        'frozen_rollouts':len(frozen),'continuations':len(report['continuations']),
        'verified_continuation_transitions':sum(r['decisions'] for r in train),
        'matched_vs_normal':matched,'revision_operations':[
            {'criterion':c['criterion'],'seed':c['seed'],**c['metrics']} for c in report['continuations']],
        'limitations':report['limitations']}
    (output/'summary.json').write_text(json.dumps(summary,indent=2))
    lines=['# Resultados: revisión de árboles y diagnóstico de aprendizaje','',
        'Comparación exploratoria de continuaciones desde snapshots. Estos retornos no son puntajes de partidas completas.',
        f'Campaña completa: **{report["complete"]}**. Rollouts congelados: {len(frozen)}. Continuaciones: {len(report["continuations"])}.',
        f'Diseño efectivo: {report["config"]["continuation_decisions"]:,} decisiones nuevas por agente; '
        f'horizonte de evaluación {report["config"]["evaluation_horizon"]} decisiones; '
        f'{report["config"]["starts"]} snapshots y {len(report["config"]["seeds"])} semillas por criterio.',
        '', '## Evaluación congelada', '',
        '| Criterio | Semilla | Checkpoint | ε | Retorno medio desde snapshot | Truncadas |',
        '|---|---:|---:|---:|---:|---:|']
    if report['config']['continuation_decisions'] < 100000:
        lines.insert(4, '**Prueba de integración reducida: estos resultados verifican ejecución, no eficacia de las variantes.**')
    for r in frozen_summary:lines.append(f'|{r["criterion"]}|{r["seed"]}|{r["counter"]}|{r["epsilon"]}|{r["mean_snapshot_return"]:.2f}|{r["truncated"]}/{r["starts"]}|')
    lines.extend(['','## Continuaciones finales frente al control normal','',
        'Diferencias pareadas por los mismos arranques. Los arranques son casos diagnósticos relacionados, no semillas de entrenamiento independientes.','',
        '| Criterio | Semilla | Modalidad | Diferencia media de retorno |','|---|---:|---|---:|'])
    for r in matched:
        value='pendiente' if r['mean_difference_vs_normal'] is None else f'{r["mean_difference_vs_normal"]:+.2f}'
        lines.append(f'|{r["criterion"]}|{r["seed"]}|{r["mode"]}|{value}|')
    lines.extend(['','## Operaciones de estructura','',
        '| Criterio | Semilla | Modalidad | Divisiones activas | Crecimientos | Podas | Reemplazos |',
        '|---|---:|---|---:|---:|---:|---:|'])
    for c in report['continuations']:
        m=c['metrics'];lines.append(f'|{c["criterion"]}|{c["seed"]}|{c["mode"]}|{m["active_splits"]}|{m["grows_total"]}|{m["prunes_total"]}|{m["replacements_total"]}|')
    lines.extend(['','## Alcance y límites','',*['- '+note for note in report['limitations']]])
    (output/'resultados.md').write_text('\n'.join(lines)+'\n')
    if make_plots:plots(output,continue_rows,frozen_summary)
    print(json.dumps({k:summary[k] for k in ('complete','frozen_rollouts','continuations','verified_continuation_transitions')}))


def plots(output, continuations, frozen):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={'normal':'#111827','frozen':'#2563eb','prune':'#16a34a','prune_regrow':'#d97706','replace':'#9333ea'}
    labels={'normal':'Crecimiento normal','frozen':'Estructura fija','prune':'Poda',
            'prune_regrow':'Poda y recrecimiento','replace':'Reemplazo local'}
    groups=sorted({(r['criterion'],r['seed']) for r in continuations})
    cols=min(3,max(1,len(groups)));nrows=max(1,(len(groups)+cols-1)//cols)
    fig,axes=plt.subplots(nrows,cols,figsize=(5*cols,4*nrows),sharex=True,squeeze=False)
    for ax,group in zip(axes.flat,groups):
        for mode,color in colors.items():
            rows=sorted([r for r in continuations if (r['criterion'],r['seed'])==group and r['mode']==mode],key=lambda r:r['point'])
            ax.plot([r['point']/1000 for r in rows],[r['mean_snapshot_return'] for r in rows],'-o',label=labels[mode],color=color,ms=3)
        ax.set(title=f'{group[0]} · {group[1]}',xlabel='Miles de decisiones adicionales',ylabel='Retorno medio desde snapshot')
        ax.grid(alpha=.2)
    for ax in list(axes.flat)[len(groups):]:ax.set_visible(False)
    fig.suptitle('Continuaciones con TD activo · evaluación sin aprendizaje desde snapshots comunes')
    axes[0,0].legend(fontsize=8);fig.tight_layout();fig.savefig(output/'continuaciones.png',dpi=170);plt.close(fig)
    groups=sorted({(r['criterion'],r['seed']) for r in frozen})
    if groups:
        cols=min(3,len(groups));nrows=(len(groups)+cols-1)//cols
        fig,axes=plt.subplots(nrows,cols,figsize=(5*cols,4*nrows),sharex=True,squeeze=False)
        epscolors={0.:'#111827',.01:'#2563eb',.1:'#d97706'}
        for ax,group in zip(axes.flat,groups):
            for epsilon,color in epscolors.items():
                rows=sorted([r for r in frozen if (r['criterion'],r['seed'])==group and r['epsilon']==epsilon],key=lambda r:r['counter'])
                if rows:ax.plot([r['counter']/1000 for r in rows],[r['mean_snapshot_return'] for r in rows],'-o',label=f'ε={epsilon}',color=color)
            ax.set(title=f'{group[0]} · {group[1]}',xlabel='Miles de decisiones del checkpoint',ylabel='Retorno medio desde snapshot')
            ax.grid(alpha=.2)
        for ax in list(axes.flat)[len(groups):]:ax.set_visible(False)
        fig.suptitle('Checkpoints sin aprendizaje · sensibilidad a la exploración en arranques distintos')
        axes[0,0].legend(fontsize=8);fig.tight_layout();fig.savefig(output/'politicas_congeladas.png',dpi=170);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--no-plots',action='store_true')
    args=p.parse_args();build(args.campaign,args.output,make_plots=not args.no_plots)
