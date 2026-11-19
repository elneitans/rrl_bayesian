"""Read-only, bounded-memory analysis of committed RRTL baseline training log chains.

Usage: python analyze_logs.py --root PATH_TO_CAMPAIGN --output OUTPUT
No model deserialization. CSVs preserve seed as the experimental unit.
"""
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path
import statistics


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def checked_json(path):
    data = json.loads(path.read_text())
    assert data['schema'] == 'pong_analysis_unit_v2', path
    assert canonical_hash(data['payload']) == data['sha256'], path
    return data['payload']


def chain(folder, head):
    refs, seen, next_start = [], set(), None
    while head:
        path = folder / head['path']
        assert path.resolve().parent == (folder / 'logs').resolve(), path
        assert head['sha256'] not in seen, 'cycle'
        seen.add(head['sha256'])
        h = hashlib.sha256()
        with path.open('rb') as f:
            header_line = f.readline()
            h.update(header_line)
            for block in iter(lambda: f.read(1024 * 1024), b''):
                h.update(block)
        assert h.hexdigest() == head['sha256'], path
        header = json.loads(header_line)
        assert header['schema'] == 'pong_log_segment_v2', path
        assert header['end'] == head['end'], path
        assert header['start'] <= header['end'], path
        assert next_start is None or header['end'] == next_start - 1, path
        refs.append((path, header))
        next_start = header['start']
        head = header['previous']
    assert not refs or next_start == 1, 'chain does not begin at 1'
    return list(reversed(refs))


def rows(refs):
    for path, header in refs:
        count = 0
        with path.open() as f:
            next(f)
            for n, line in enumerate(f, header['start']):
                row = json.loads(line)
                assert row['interaction'] == n and row['transition_id'] == n - 1, path
                count += 1
                yield row
        assert count == header['end'] - header['start'] + 1, path


EP_FIELDS = ['seed', 'episode_id', 'episode_seed', 'end_interaction', 'length',
             'raw_return', 'terminated', 'truncated', 'epsilon_end', 'fit_id_end']
FIT_FIELDS = ['seed', 'interaction', 'fit_id', 'target_model_id', 'action', 'updated',
              'samples', 'data_hash', 'max_abs_q', 'targets_outside_scale',
              'elapsed_seconds', 'm', 'burn_in', 'retained', 'sweeps',
              'grow_acceptance', 'prune_acceptance', 'grow_accepted', 'prune_accepted',
              'grow_impossible', 'prune_impossible']
WIN_FIELDS = ['seed', 'bin', 'start_interaction', 'end_interaction', 'decisions',
              'raw_reward_sum', 'reward_sum', 'ended_episodes', 'mean_episode_return',
              'positive_return_fraction', 'mean_epsilon', 'fire_fraction',
              'noop_fraction', 'right_fraction', 'left_fraction']


def analyze(root, output, bin_size=10000, seeds=None):
    output.mkdir(parents=True, exist_ok=True)
    statuses = sorted(p for p in root.rglob('status.json')
                      if p.parent.parent.name == 'corrected_common' and p.parent.name.isdigit())
    if seeds:
        selected = set(seeds)
        statuses = [p for p in statuses if int(p.parent.name) in selected]
        assert {int(p.parent.name) for p in statuses} == selected, 'Missing requested seeds'
    assert statuses, f'No RRTL baseline seed folders in {root}'
    summary = {'scope': 'Committed training logs only; not greedy evaluation curves',
               'bin_size': bin_size, 'seeds': [], 'diagnostic_notes': [
                   'Episode returns are training returns under changing epsilon.',
                   'Only ended episodes enter episodes.csv, including marked truncations.',
                   'Each bin assigns an episode to the decision where it ends.',
                   'Acceptance rates are not convergence diagnostics.',
                   'max_abs_q is absent in some logger versions; missing is not zero.']}
    with (output / 'episodes.csv').open('w', newline='') as ef, \
         (output / 'fits.csv').open('w', newline='') as ff, \
         (output / 'windows.csv').open('w', newline='') as wf:
        ew, fw, ww = [csv.DictWriter(f, fieldnames=fields) for f, fields in
                       [(ef, EP_FIELDS), (ff, FIT_FIELDS), (wf, WIN_FIELDS)]]
        for w in (ew, fw, ww):
            w.writeheader()
        for sp in statuses:
            status = checked_json(sp)
            seed = int(sp.parent.name)
            refs = chain(sp.parent, status['log_head'])
            recent, first = collections.deque(maxlen=100), []
            bins = collections.defaultdict(lambda: {'count': 0, 'raw': 0., 'reward': 0.,
                'eps': 0., 'actions': collections.Counter(), 'returns': []})
            episodes, fits, n, raw_total = 0, 0, 0, 0.
            min_q, max_q, q_present, target_outside, fit_seconds = None, None, 0, 0, 0.
            last_episode_id = -1
            for row in rows(refs):
                n = row['interaction']
                bucket = bins[(n - 1) // bin_size]
                bucket['count'] += 1
                bucket['raw'] += row['raw_reward']
                raw_total += row['raw_reward']
                bucket['reward'] += row['reward']
                bucket['eps'] += row['epsilon']
                bucket['actions'][row['action']] += 1
                if row.get('episode_end', row['terminated'] or row['truncated']):
                    assert row['episode_id'] > last_episode_id, 'duplicate ended episode'
                    last_episode_id = row['episode_id']
                    ret = row['raw_episode_return']
                    bucket['returns'].append(ret)
                    recent.append(ret)
                    if len(first) < 100:
                        first.append(ret)
                    episodes += 1
                    ew.writerow(dict(seed=seed, episode_id=row['episode_id'],
                        episode_seed=row['episode_seed'], end_interaction=n,
                        length=row['step_id'] + 1, raw_return=ret,
                        terminated=row['terminated'], truncated=row['truncated'],
                        epsilon_end=row['epsilon'], fit_id_end=row['fit_id']))
                if row.get('fit'):
                    fits += 1
                    for action, d in row['fit']['actions'].items():
                        moves = d.get('moves', {})
                        values = {k: d.get(k) for k in FIT_FIELDS if k in d}
                        values.update(seed=seed, interaction=n, fit_id=row['fit_id'],
                            target_model_id=row['fit']['target_model_id'], action=action)
                        for move in ('grow', 'prune'):
                            m = moves.get(move, {})
                            values[move + '_acceptance'] = m.get('acceptance_given_possible')
                            values[move + '_accepted'] = m.get('accepted')
                            values[move + '_impossible'] = m.get('impossible')
                        fw.writerow(values)
                        fit_seconds += d.get('elapsed_seconds', 0.)
                        target_outside += d.get('targets_outside_scale', 0)
                        q = d.get('max_abs_q')
                        if q is not None:
                            q_present += 1
                            min_q = q if min_q is None else min(min_q, q)
                            max_q = q if max_q is None else max(max_q, q)
            assert n == status['counter'], (seed, n, status['counter'])
            assert episodes == status['metrics']['ended_episodes'], (seed, episodes)
            for b, v in sorted(bins.items()):
                rs, count = v['returns'], v['count']
                ww.writerow(dict(seed=seed, bin=b, start_interaction=b*bin_size+1,
                    end_interaction=min((b+1)*bin_size,n), decisions=count,
                    raw_reward_sum=v['raw'], reward_sum=v['reward'], ended_episodes=len(rs),
                    mean_episode_return=statistics.mean(rs) if rs else None,
                    positive_return_fraction=statistics.mean(r > 0 for r in rs) if rs else None,
                    mean_epsilon=v['eps']/count, **{a+'_fraction':v['actions'][i]/count
                        for i,a in enumerate(('noop','fire','right','left'))}))
            record = dict(seed=seed, counter=n, episodes=episodes, fits=fits,
                log_segments=len(refs), first100_n=len(first), last100_n=len(recent),
                first100_mean=statistics.mean(first) if first else None,
                last100_mean=statistics.mean(recent) if recent else None,
                first100_positive_fraction=statistics.mean(r>0 for r in first) if first else None,
                last100_positive_fraction=statistics.mean(r>0 for r in recent) if recent else None,
                raw_reward_total=raw_total, fit_seconds_sum_action_diagnostics=fit_seconds,
                targets_outside_scale_sum=target_outside, max_abs_q_observations=q_present,
                max_abs_q_min=min_q, max_abs_q_max=max_q)
            summary['seeds'].append(record)
            print(json.dumps(record), flush=True)
            for f in (ef,ff,wf):
                f.flush()
    (output / 'training_summary.json').write_text(json.dumps(summary, indent=2))
    return summary


def plot(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    def read_csv(name):
        with (output / name).open() as f:
            return list(csv.DictReader(f))
    eps, wins, fits = [read_csv(name) for name in ('episodes.csv','windows.csv','fits.csv')]
    seeds = sorted({int(r['seed']) for r in wins})
    plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False})
    fig, axes = plt.subplots(2,5, figsize=(18,7), sharex=False, sharey=True)
    global_max = max([float(r['raw_return']) for r in eps] or [45.])
    for ax, seed in zip(axes.flat, seeds):
        e = [r for r in eps if int(r['seed']) == seed]
        x = np.array([int(r['episode_id'])+1 for r in e])
        y = np.array([float(r['raw_return']) for r in e])
        ax.scatter(x,y,s=3,alpha=.12,color='#4477aa')
        if len(y)>=100:
            ax.plot(x[99:],np.convolve(y,np.ones(100)/100,mode='valid'),color='#cc3311',lw=1.8)
        ax.set_title(str(seed));ax.set_xlabel('Partida de entrenamiento');ax.set_ylabel('Retorno bruto')
        ax.set_ylim(0,max(45,global_max)*1.03)
        ax.grid(alpha=.15)
    fig.suptitle('RRTL baseline Breakout · retorno de entrenamiento; rojo = media móvil de 100 partidas')
    fig.tight_layout();fig.savefig(output/'training_returns.png',dpi=170);plt.close(fig)
    fig, axes=plt.subplots(2,1,figsize=(12,9),sharex=True)
    common_end=min(max(int(r['end_interaction']) for r in wins if int(r['seed'])==s) for s in seeds)
    grouped=collections.defaultdict(list)
    for seed in seeds:
        w=[r for r in wins if int(r['seed'])==seed and r['mean_episode_return']]
        x=[int(r['end_interaction']) for r in w]
        axes[0].plot(x,[float(r['mean_episode_return']) for r in w],lw=.9,alpha=.5,label=str(seed))
        axes[1].plot(x,[float(r['fire_fraction']) for r in w],lw=.9,alpha=.6)
        for r in w:
            if int(r['decisions'])==10000 and int(r['end_interaction'])<=common_end:
                grouped[int(r['end_interaction'])].append(float(r['mean_episode_return']))
    gx=sorted(x for x,v in grouped.items() if len(v)==len(seeds))
    gy=[np.mean(grouped[x]) for x in gx]
    axes[0].plot(gx,gy,color='black',lw=2.3,label=f'Media entre {len(seeds)} semillas (bins comunes)')
    axes[0].legend(ncol=4,fontsize=8);axes[0].set_ylabel('Retorno medio por partida terminada')
    axes[1].set_ylabel('Fracción de acciones FIRE');axes[1].set_xlabel('Decisiones acumuladas')
    for ax in axes:ax.grid(alpha=.2);ax.axvline(50000,color='grey',ls=':',alpha=.8)
    fig.suptitle('Ventanas de 10.000 decisiones · entrenamiento con exploración; no evaluación greedy')
    fig.tight_layout();fig.savefig(output/'training_binned.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(3,1,figsize=(12,10),sharex=True)
    for seed in seeds:
        f=[r for r in fits if int(r['seed'])==seed]
        by=collections.defaultdict(list)
        for r in f:by[int(r['interaction'])].append(r)
        xs=sorted(by)
        ys=[[sum(float(r.get(key) or 0) for r in by[x]) for x in xs]
            for key in ('elapsed_seconds','targets_outside_scale')]
        axes[0].plot(xs,ys[0],alpha=.4,lw=.6)
        axes[1].plot(xs,ys[1],alpha=.4,lw=.6)
        q=[(int(r['interaction']),float(r['max_abs_q'])) for r in f if r['max_abs_q']]
        if q:axes[2].scatter(*zip(*q),s=2,alpha=.2)
    axes[0].set_ylabel('Segundos por ajuste (suma acciones)')
    axes[1].set_ylabel('Objetivos fuera de escala (conteo)')
    axes[2].set_ylabel('Máximo |Q| en lote por acción');axes[2].set_xlabel('Decisiones acumuladas')
    if not any(r['max_abs_q'] for r in fits):
        axes[2].text(.5,.5,'El logger de esta campaña no registra max_abs_q; requiere checkpoints',
            transform=axes[2].transAxes,ha='center')
    fig.suptitle('Diagnósticos de ajustes RRTL baseline · aceptación no demuestra convergencia')
    fig.tight_layout();fig.savefig(output/'fit_diagnostics.png',dpi=170);plt.close(fig)
    summary_path=output/'training_summary.json'
    if summary_path.exists():
        summary=json.loads(summary_path.read_text())
        pairs=[r for r in summary['seeds'] if r['first100_n']==100 and r['last100_n']==100]
        before=np.array([r['first100_mean'] for r in pairs])
        after=np.array([r['last100_mean'] for r in pairs])
        differences=after-before
        rng=np.random.default_rng(20261101)
        boot=np.mean(rng.choice(differences,size=(20000,len(differences)),replace=True),axis=1)
        comparison={'seeds':len(pairs),'mean_first100':float(before.mean()),
            'mean_last100':float(after.mean()),'mean_paired_difference':float(differences.mean()),
            'paired_seed_bootstrap_95_percentile':np.quantile(boot,[.025,.975]).tolist(),
            'seeds_improved':int(np.sum(differences>0)),
            'note':'Descriptive training comparison, not random-policy baseline or greedy evaluation. Epsilon and visitation change. Bootstrap resamples training seeds, not episodes.'}
        (output/'first_last_comparison.json').write_text(json.dumps(comparison,indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--plot-only',action='store_true')
    ap.add_argument('--seeds',type=int,nargs='+',help='Explicit subset; missing seeds fail')
    args=ap.parse_args()
    if not args.plot_only:
        assert args.root,'--root required'
        analyze(args.root,args.output,seeds=args.seeds)
    plot(args.output)
