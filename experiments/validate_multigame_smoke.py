"""Bounded C15 reproduction using B's single-cell runner, not a separate learning pipeline."""
import argparse
import json
from pathlib import Path

from bayesian_rrtl.online.protocol import freeze
from bayesian_rrtl.online.storage import binding,load_frontier
from experiments.compare_pong_progressive import train_cell,evaluate_cell,consolidate,StopRequest
from experiments.pong_analysis_runtime import checked_read,atomic_json
from experiments.report_atari_comparison import export


def run(protocol, output, *, pause_at=19):
    protocol=Path(protocol)
    raw=json.loads(protocol.read_text())
    if raw['schema']!='atari_comparison_v3' or raw['level']!='smoke':
        raise ValueError('Only explicitly small multigame smoke is accepted')
    if any(g['max_decisions']>256 for g in raw['games'].values()):
        raise ValueError('Smoke wrapper cannot dispatch long training')
    manifest=freeze(raw,output)
    with StopRequest() as stop:
        for index,cell in enumerate(manifest['cells']):
            if stop.requested:
                return {'state':'partial','reason':'signal'}
            first=train_cell(output,index,decision_limit=pause_at,stop=stop)
            if first['state']!='completed':
                final=train_cell(output,index,stop=stop)
            else:
                final=first
            if final['state']!='completed':
                return {'state':'partial','cell':cell['id'],'reason':final['reason']}
            if cell['learner'] in ('cart','bart','extra_trees') and final['metrics']['learner']['fits']<2:
                raise ValueError('Technical smoke must perform at least two actual fits')
            folder=Path(output)/cell['id']
            data,_=load_frontier(folder,final,binding(manifest,cell))
            if data['counter']!=raw['games'][cell['game']]['max_decisions']:
                raise ValueError('Smoke counter mismatch')
            results=evaluate_cell(output,index,stop=stop)
            if any(r['state']!='completed' for r in results):
                return {'state':'partial','cell':cell['id'],'reason':'evaluation_partial'}
            # Validate immutable completed session is reused, and greedy evaluation does not mutate it.
            before=checked_read(folder/'status.json')
            assert train_cell(output,index,stop=stop)==before
            print(cell['id'],'complete',flush=True)
    report=consolidate(output)
    export(output)
    result={'state':'completed' if report['completed'] else 'partial',
            'cells':len(manifest['cells']),'fits_verified':True,'reserved_test_executed':False}
    atomic_json(Path(output)/'reports'/'smoke_validation.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol',type=Path,default=Path('configs/atari_comparison_smoke.json'))
    p.add_argument('--campaign',type=Path,required=True)
    args=p.parse_args()
    result=run(args.protocol,args.campaign)
    print(json.dumps(result,indent=2))
    return 0 if result['state']=='completed' else 3


if __name__=='__main__':
    raise SystemExit(main())
