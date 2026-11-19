"""Read committed observations during the run; never asserts model completion."""
import argparse
import collections
import json
from pathlib import Path
import statistics

from build_report import checked,plots,save_csv,trace_summary
from partial_frozen import digest,audit as audit_frozen


def audit(root,output,*,make_plots=True):
    output.mkdir(parents=True,exist_ok=True)
    manifest=checked(root/'manifest.json');config=manifest['config'];campaign=digest(manifest)
    snapshots=checked(root/'starts.json');starts={r['id']:r for r in snapshots}
    inventory=[];evaluations=[];training=[];events=[]
    for group in manifest['groups']:
        parent=next(p for p in manifest['parents'] if p['criterion']==group['criterion'] and p['seed']==group['seed'] and p['counter']==config['parent_counter'])
        for mode in config['modes']:
            folder=root/'continuations'/group['criterion']/str(group['seed'])/mode
            base={**group,'mode':mode}
            if not (folder/'status.json').exists():
                inventory.append({**base,'state':'not_started','continuation_decisions':0});continue
            status=checked(folder/'status.json')
            identity={'campaign':campaign,'parent':parent['checkpoint'],**base,'parent_counter':config['parent_counter']}
            if status['identity']!=identity or status['counter']!=config['parent_counter']+status['continuation_decisions']:
                raise ValueError('Continuation frontier/identity mismatch')
            inventory.append({**base,'state':status['state'],'continuation_decisions':status['continuation_decisions'],**status['metrics']})
            training.extend({**base,**row} for row in trace_summary(folder,status,config['parent_counter']))
            for point in range(0,status['continuation_decisions']+1,config['evaluation_interval']):
                results=[]
                for start in snapshots:
                    path=folder/'evaluations'/str(point)/str(start['id'])/'state.json'
                    if not path.exists():continue
                    state=checked(path);ei=checked(path.with_name('identity.json'))
                    expected={**identity,'checkpoint':status['checkpoints'][str(point)],'point':point,
                        'start_bank':manifest['start_bank'],'start':start['id'],'epsilon':0.,
                        'horizon':config['evaluation_horizon'],'evaluation_seed':config['evaluation_seed']+start['id']}
                    if ei!=expected or state['identity_sha256']!=digest(ei):raise ValueError('Continuation evaluation binding mismatch')
                    if state['status']!='completed':continue
                    result=state['result']
                    if result['physical_hash']!=starts[result['start']]['physical_hash']:
                        raise ValueError('Continuation evaluation start mismatch')
                    if result['seed']!=expected['evaluation_seed'] or result['epsilon']!=0. or result['horizon']!=expected['horizon']:
                        raise ValueError('Continuation evaluation settings mismatch')
                    results.append(result)
                evaluations.append({**base,'point':point,'starts':len(results),'expected_starts':len(starts),
                    'mean_snapshot_return':statistics.mean(r['raw_return'] for r in results) if len(results)==len(starts) else None,
                    'truncated':sum(r['truncated'] for r in results)})
            for path in (folder/'audits').glob('*.json'):
                if path.stem not in status['checkpoints']:continue
                saved=checked(path)
                if saved['identity']!=identity:raise ValueError('Audit binding mismatch')
                events.extend({**base,**event} for event in saved['events'])
    save_csv(output/'continuation_available_inventory.csv',inventory)
    save_csv(output/'continuation_available_evaluations.csv',evaluations)
    save_csv(output/'continuation_available_training.csv',training)
    (output/'continuation_available_events.json').write_text(json.dumps(events,indent=2)+'\n')
    frozen=audit_frozen(root,output)
    result={'scope':'Observed committed traces and evaluation JSON only; final models/holdouts not audited here',
        'reported_completed':sum(r['state']=='completed' for r in inventory),'expected_continuations':len(inventory),
        'verified_trace_transitions':sum(r['decisions'] for r in training),'frozen_available':frozen['available'],
        'inventory':inventory,'evaluations':evaluations}
    (output/'continuation_partial.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# Observaciones parciales de las continuaciones','',result['scope'],
        f'Fronteras reportadas completas: {result["reported_completed"]}/{len(inventory)}. Transiciones de trazas verificadas: {result["verified_trace_transitions"]:,}.',
        'Los valores se refieren a estados descargados; las tareas pueden haber avanzado después. No se declara completitud de la campaña.','',
        '| Criterio | Semilla | Modalidad | Decisiones nuevas | Estado |','|---|---:|---|---:|---|']
    for row in inventory:lines.append(f'|{row["criterion"]}|{row["seed"]}|{row["mode"]}|{row["continuation_decisions"]}|{row["state"]}|')
    lines.extend(['','## Evaluaciones densas disponibles','',
        'Solo se promedian puntos con los seis snapshots completos; faltantes permanecen pendientes.','',
        '| Criterio | Semilla | Modalidad | Decisiones nuevas | Retorno desde snapshot |','|---|---:|---|---:|---:|'])
    for row in evaluations:
        val='pendiente' if row['mean_snapshot_return'] is None else f'{row["mean_snapshot_return"]:.2f}'
        lines.append(f'|{row["criterion"]}|{row["seed"]}|{row["mode"]}|{row["point"]}|{val}|')
    (output/'observaciones.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:result[k] for k in ('reported_completed','expected_continuations','verified_trace_transitions','frozen_available')}))
    if make_plots:plots(output,evaluations,frozen['summary'])
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--no-plots',action='store_true')
    args=p.parse_args();audit(args.campaign,args.output,make_plots=not args.no_plots)
