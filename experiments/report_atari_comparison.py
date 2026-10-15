"""Regenerate C15 figures, inventory and faithful model exports; never evaluates a policy."""
import argparse
from pathlib import Path

from bayesian_rrtl.evaluation import export_interpretability
from bayesian_rrtl.online.protocol import load
from bayesian_rrtl.online.storage import binding,load_frontier
from experiments.compare_pong_progressive import consolidate
from experiments.pong_analysis_runtime import atomic_json,checked_read,bytes_hash


def export(output):
    output=Path(output)
    manifest=load(output)
    if manifest['schema'] != 'atari_comparison_v3':
        raise ValueError('Multigame report required')
    report=consolidate(output)
    artifacts=[]
    for cell in manifest['cells']:
        folder=output/cell['id']
        if not (folder/'status.json').exists():
            continue
        status=checked_read(folder/'status.json')
        if not status.get('checkpoint'):
            continue
        payload,_=load_frontier(folder,status,binding(manifest,cell))
        agent=payload['session']['collector']['agent']
        state=payload['session']['collector']['state']
        references=[[],state or []]
        rules=export_interpretability(agent,cell['learner'],references)
        rules.update(cell=cell['id'],checkpoint=status['checkpoint'],
                     references='empty and current training observation; not a separate fidelity dataset')
        path=output/'reports'/'models'/(cell['id'].replace('/','_')+'.json')
        atomic_json(path,rules)
        artifacts.append({'path':str(path.relative_to(output)),'sha256':bytes_hash(path.read_bytes()),
                          'checkpoint':status['checkpoint'],'cell':cell['id']})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    analysis=report['analysis']
    for game in manifest['protocol']['games']:
        for axis in ('decisions','active_seconds'):
            fig,ax=plt.subplots(figsize=(7,4))
            for run in analysis['runs']:
                if run['game'] != game or not run['curve']:
                    continue
                x=[p['interactions'] if axis=='decisions' else p['training_active_seconds'] for p in run['curve']]
                if any(v is None for v in x):
                    continue
                ax.plot(x,[p['mean_return'] for p in run['curve']],marker='.',
                        label=f"{run['pipeline']}/{run['learner']}/{run['seed']}")
            ax.set(xlabel='Training decisions' if axis=='decisions' else 'Measured training active seconds',
                   ylabel='Development raw return (scope composition in JSON)',title=game)
            if ax.lines:
                ax.legend(fontsize='x-small')
            fig.tight_layout()
            path=output/'reports'/f'{game}_{axis}.png'
            fig.savefig(path,dpi=150)
            plt.close(fig)
            artifacts.append({'path':str(path.relative_to(output)),'sha256':bytes_hash(path.read_bytes())})
    inventory={'campaign':report['campaign'],'artifacts':artifacts,
               'runs':[{k:v for k,v in r.items() if k not in ('evaluations','metrics')} for r in report['cells']]}
    atomic_json(output/'reports'/'inventory.json',inventory)
    return inventory


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign',type=Path,required=True)
    args=p.parse_args()
    result=export(args.campaign)
    print('Regenerated artifacts:',len(result['artifacts']))


if __name__=='__main__':
    main()
