"""Read-only audit of trusted completed Breakout campaigns; no ALE execution.

Run from repository root with the pinned .venv. Checkpoints are verified before
unpickling and are used for static predictions only, never resumed here.
"""
import argparse
import collections
import csv
import hashlib
import importlib.util
import json
import pickle
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from preprocessing import Fact
from bayesian_rrtl.encoding import CATEGORIES
from bayesian_rrtl.baseline_anova import anova_from_stats

spec = importlib.util.spec_from_file_location('verified_logs', REPO / 'analysis/baseline_10_seeds_20261102/analyze_logs.py')
loglib = importlib.util.module_from_spec(spec)
spec.loader.exec_module(loglib)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def read(path):
    return loglib.checked_json(path)


def load(path, counter, binding):
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == path.stem.split('-', 1)[1], path
    meta = read(Path(str(path) + '.json'))
    payload = pickle.loads(raw)  # Only own hash-verified project artifacts.
    assert payload['counter'] == meta['counter'] == counter, path
    assert payload['binding'] == meta['binding'] == binding, path
    return payload


def save_csv(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)


def probe_state(agent, position, motion):
    state = []
    for key in agent.encoder.catalog:
        value = 'True' if key.name == 'present' else 'False' if key.name == 'incontact' else 'same'
        if key.name == 'x' and key.obj1 == 'player_t' and key.obj2 == 'ball_t':
            value = position
        if key.name == 'x' and key.obj1 == 'player_t' and key.obj2 == 'player_t-1':
            value = motion
        if key.name == 'y' and key.obj1 == 'player_t' and key.obj2 == 'ball_t':
            value = 'more'
        if key.obj1 == 'ball_t' and key.obj2 == 'ball_t-1':
            value = 'more'
        state.append(Fact(key.type, value, key.name, key.obj1, key.obj2, key.obj3))
    return tuple(state)


def inspect_campaign(root, label, out, process_logs):
    manifest = read(root / 'campaign.json')
    campaign_hash = digest(manifest)
    cells = manifest['cells']
    assert len(cells) == 10 and sorted(c['seed'] for c in cells) == list(range(710601, 710611))
    evals, structures, nodes, probes, changes, summaries, candidates = [], [], [], [], [], [], []
    texts, bins, episodes = [], [], []
    for cell in cells:
        folder = root / cell['id']
        seed = cell['seed']
        status = read(folder / 'status.json')
        binding = status['binding']
        assert binding['campaign'] == campaign_hash and binding['cell'] == cell['hash']
        assert status['counter'] == 1000000 and binding['runtime'] == manifest['runtime']
        assert status['reason'] == 'authorized_milestone_reached'
        expected = set()
        schedule = manifest['protocol']['games']['Breakout']['evaluation']
        for group in ('intermediate', 'final_development'):
            for counter in schedule[group]['checkpoints']:
                if counter <= 1000000:
                    for es in schedule[group]['seeds']:
                        expected.add((group, counter, es))
        seen = set()
        for path in sorted((folder / 'evaluations').glob('*/*/*/state.json')):
            state = read(path)
            identity = read(path.with_name('identity.json'))
            assert state['identity_sha256'] == digest(identity)
            assert identity['campaign'] == campaign_hash and identity['cell'] == cell['hash']
            group, counter, es = path.parts[-4], int(path.parts[-3]), int(path.parts[-2])
            assert identity['group'] == group and identity['seed'] == es
            assert identity['runtime'] == manifest['runtime']
            assert identity['policy'] == 'greedy_uniform_ties_v1'
            assert identity['horizon'] == schedule[group]['horizon']
            assert state['result']['seed'] == es
            assert identity['checkpoint']['path'] == f'checkpoints/{counter}-{identity["checkpoint"]["sha256"]}.pkl'
            assert (folder / identity['checkpoint']['path']).is_file()
            assert identity['checkpoint'] == status['checkpoint'] if counter == 1000000 else True
            assert state['status'] == 'completed'
            key = (group, counter, es)
            assert key in expected and key not in seen
            seen.add(key)
            evals.append(dict(method=label, seed=seed, group=group, counter=counter,
                              eval_seed=es, checkpoint_sha256=identity['checkpoint']['sha256'],
                              **{k: v for k, v in state['result'].items() if k != 'seed'}))
        assert seen == expected, (seed, expected - seen)
        previous = None
        paths = sorted(folder.glob('checkpoints/*.pkl'), key=lambda p: int(p.name.split('-')[0]))
        retained = {int(p.name.split('-')[0]) for p in paths}
        assert {0, 100000, 500000, 1000000} <= retained
        for path in paths:
            counter = int(path.name.split('-')[0])
            payload = load(path, counter, binding)
            agent = payload['session']['collector']['agent']
            assert agent.interactions == counter
            localnodes, rules = [], {}
            def walk(node, action, route='root'):
                record = dict(method=label, seed=seed, counter=counter, action=action,
                              path=route, depth=node.depth, literal=str(node.literal), q=node.q_value,
                              leaf=node.literal is None)
                localnodes.append(record)
                if node.literal is not None:
                    rules[(action, route)] = str(node.literal)
                    labels = ('more', 'same', 'less') if node.literal.type == 'comparative' else ('true', 'false')
                    for name in labels:
                        walk(getattr(node, name + '_branch'), action, route + '/' + name)
                elif counter == 1000000:
                    for lit, stats in node.refinements_stats.items():
                        a = anova_from_stats(stats)
                        total = stats['Total']
                        within = sum(s['J'] for k, s in stats.items() if k != 'Total')
                        from scipy.stats import f
                        old_p = float(f.cdf(within / total['J'], a.n - 1, a.n - 1)) if total['J'] > 0 and a.n > 1 else None
                        candidates.append(dict(method=label, seed=seed, action=action, path=route,
                            literal=str(lit), depth=node.depth, n=a.n, groups=a.k,
                            within=a.within, between=a.between,
                            effect_fraction=a.between / (a.within + a.between) if a.within + a.between else 0.,
                            p_historical=old_p, p_anova=a.p_value,
                            eligible_depth=node.depth < node.max_depth, eligible_n=a.n > node.min_sample_size,
                            category_counts=json.dumps({k: v['n'] for k, v in stats.items() if k != 'Total'})))
            for action, node in agent.root_nodes.items():
                walk(node, action)
            nodes.extend(localnodes)
            leaves = [n for n in localnodes if n['leaf']]
            panel_path = REPO / f'analysis/bart_10_seeds_20261101/panel_{seed}.npz'
            values = np.load(panel_path)['values']
            panel = [tuple(Fact(k.type, CATEGORIES[k.type][int(v)], k.name, k.obj1, k.obj2, k.obj3)
                           for k, v in zip(agent.encoder.catalog, row) if v >= 0) for row in values]
            q = np.array([[root.predict(s) for root in agent.root_nodes.values()] for s in panel])
            structures.append(dict(method=label, seed=seed, counter=counter, splits=agent.n_splits,
                leaves=len(leaves), max_depth=max(n['depth'] for n in localnodes),
                leaves_at_depth_limit=sum(n['depth'] == 4 for n in leaves),
                q_min=float(q.min()), q_max=float(q.max()), checkpoint_sha256=path.stem.split('-', 1)[1]))
            if previous:
                pc, pq, pr = previous
                preserved = all(rules.get(k) == v for k, v in pr.items())
                assert preserved
                flips = ~np.any((q == q.max(1, keepdims=True)) & (pq == pq.max(1, keepdims=True)), axis=1)
                changes.append(dict(method=label, seed=seed, start=pc, end=counter,
                    new_splits=len(rules) - len(pr), q_rmse=float(np.sqrt(np.mean((q - pq) ** 2))),
                    disjoint_argmax_pct=float(100 * flips.mean()), panel_states=len(q), old_rules_preserved=preserved))
            previous = counter, q, rules
            if counter in (100000, 500000, 1000000):
                texts.append(f'\n{label} SEED {seed} CHECKPOINT {counter}\n' + '\n'.join(
                    f'{n["action"]}/{n["path"]}: ' + (f'Q={n["q"]:.8f}' if n['leaf'] else n['literal']) for n in localnodes))
            for position in ('less', 'same', 'more'):
                for motion in ('less', 'same', 'more'):
                    state = probe_state(agent, position, motion)
                    qs = np.array([tree.predict(state) for tree in agent.root_nodes.values()])
                    probes.append(dict(method=label, seed=seed, counter=counter, position=position,
                        previous_player_motion=motion, greedy='/'.join(np.array(agent.actions)[qs == qs.max()]),
                        **dict(zip(agent.actions, map(float, qs)))))
        sm = dict(method=label, seed=seed, counter=status['counter'],
                  ended_episodes=status['metrics']['ended_episodes'],
                  truncated_episodes=status['metrics']['truncated_episodes'],
                  phase_seconds=status['metrics']['phase_seconds'],
                  fallback=status['metrics']['learner']['fallback'], binding=binding)
        if process_logs:
            refs = loglib.chain(folder, status['log_head'])
            bybin = collections.defaultdict(lambda: dict(n=0, raw=0., reward=0., eps=0.,
                td_sq=0., td_abs=0., target_sum=0., newq_sum=0., actions=collections.Counter(), ended=[]))
            first, last, count = [], collections.deque(maxlen=100), 0
            for row in loglib.rows(refs):
                count += 1
                v = bybin[(row['interaction'] - 1) // 10000]
                v['n'] += 1
                v['raw'] += row['raw_reward']
                v['reward'] += row['reward']
                v['eps'] += row['epsilon']
                v['actions'][row['action']] += 1
                td = row['td']; error = td['target'] - td['old_q']
                v['td_sq'] += error * error; v['td_abs'] += abs(error)
                v['target_sum'] += td['target']; v['newq_sum'] += td['new_q']
                if row['episode_end']:
                    ret = row['raw_episode_return']; last.append(ret)
                    if len(first) < 100:
                        first.append(ret)
                    v['ended'].append(ret)
                    episodes.append(dict(method=label, seed=seed, episode_id=row['episode_id'],
                        end_interaction=row['interaction'], length=row['step_id'] + 1,
                        raw_return=ret, terminated=row['terminated'], truncated=row['truncated']))
            assert count == status['counter']
            assert sum(len(v['ended']) for v in bybin.values()) == sm['ended_episodes']
            sm.update(first100_mean=float(np.mean(first)), last100_mean=float(np.mean(last)),
                      verified_segments=len(refs), verified_transitions=count)
            for b, v in sorted(bybin.items()):
                bins.append(dict(method=label, seed=seed, start=b * 10000 + 1, end=(b + 1) * 10000,
                    decisions=v['n'], raw_reward=v['raw'], transformed_reward=v['reward'],
                    ended=len(v['ended']), mean_episode_return=float(np.mean(v['ended'])) if v['ended'] else None,
                    epsilon=v['eps'] / v['n'], td_mse=v['td_sq'] / v['n'], td_mae=v['td_abs'] / v['n'],
                    mean_target=v['target_sum'] / v['n'], mean_new_q=v['newq_sum'] / v['n'],
                    **{a + '_fraction': v['actions'][i] / v['n'] for i, a in enumerate(agent.actions)}))
        summaries.append(sm)
        print(label, seed, 'trees/evaluations verified;', 'logs verified' if process_logs else 'logs not processed', flush=True)
    for name, data in [('evaluations', evals), ('structures', structures), ('nodes', nodes), ('policy_probes', probes),
                       ('policy_changes', changes), ('candidate_statistics', candidates), ('windows', bins), ('episodes', episodes)]:
        save_csv(out / f'{label}_{name}.csv', data)
    (out / f'{label}_trees.txt').write_text('\n'.join(texts))
    (out / f'{label}_summary.json').write_text(json.dumps(summaries, indent=2))
    (out / f'{label}_manifest_hash.json').write_text(json.dumps({'campaign_hash': campaign_hash}))
    return manifest


def plots(out):
    es = pd.concat([pd.read_csv(out / f'{m}_evaluations.csv') for m in ('historical', 'anova')])
    ss = pd.concat([pd.read_csv(out / f'{m}_structures.csv') for m in ('historical', 'anova')])
    colors = {'historical': '#2563eb', 'anova': '#c2410c'}
    labels = {'historical': 'Histórico', 'anova': 'ANOVA'}
    fig, axes = plt.subplots(2, 5, figsize=(16, 6.5), sharex=True, sharey=True)
    for ax, seed in zip(axes.flat, range(710601, 710611)):
        for method, color in colors.items():
            d = es[(es.method == method) & (es.seed == seed) & (es.group == 'intermediate')].groupby('counter').raw_return.mean()
            ax.plot(d.index / 1000, d.values, '-o', color=color, label=labels[method], ms=4)
        ax.set(title=str(seed), xlabel='Miles de decisiones', ylabel='Retorno bruto greedy')
        ax.grid(alpha=.2)
    axes[0, 0].legend()
    fig.suptitle('Breakout · baseline histórico frente a ANOVA · puntos observados, no evaluación continua')
    fig.tight_layout(); fig.savefig(out / 'greedy_por_semilla.png', dpi=180); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for method, color in colors.items():
        d = es[(es.method == method) & (es.group == 'intermediate')].groupby(['seed', 'counter']).raw_return.mean().unstack('seed')
        axes[0].plot(d.index / 1000, d.mean(axis=1), '-o', color=color, label=labels[method])
        s = ss[ss.method == method].groupby('counter').splits.mean()
        s = s[s.index.isin([0, 100000, 500000, 1000000])]
        axes[1].plot(s.index / 1000, s.values, '-o', color=color, label=labels[method])
    axes[0].set(xlabel='Miles de decisiones', ylabel='Media entre 10 agentes', title='Greedy: media descriptiva')
    axes[1].set(xlabel='Miles de decisiones', ylabel='Divisiones totales en cuatro árboles', title='Crecimiento estructural medio')
    for ax in axes: ax.legend(); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(out / 'comparacion_global.png', dpi=180); plt.close(fig)
    if any((out / f'{m}_windows.csv').stat().st_size < 10 for m in colors):
        return  # Static checkpoint/evaluation audit may precede log download.
    allwin = pd.concat([pd.read_csv(out / f'{m}_windows.csv') for m in colors])
    fig, axes = plt.subplots(2, 5, figsize=(16, 6.5), sharex=True, sharey=True)
    for ax, seed in zip(axes.flat, range(710601, 710611)):
        for method, color in colors.items():
            d = allwin[(allwin.seed == seed) & (allwin.method == method)]
            ax.plot(d.end / 1000, d.mean_episode_return.rolling(5, min_periods=1).mean(), color=color, label=labels[method])
        ax.set(title=str(seed), xlabel='Miles de decisiones', ylabel='Retorno train por episodio')
        ax.grid(alpha=.2)
    axes[0, 0].legend()
    fig.suptitle('Entrenamiento con exploración: medias de ventanas10k, suavizadas con5 ventanas; no greedy')
    fig.tight_layout(); fig.savefig(out / 'entrenamiento_por_semilla.png', dpi=180); plt.close(fig)


def summarize_training(out):
    summaries, blocks = {}, []
    for method in ('historical', 'anova'):
        seeds = json.loads((out / f'{method}_summary.json').read_text())
        episodes = pd.read_csv(out / f'{method}_episodes.csv')
        windows = pd.read_csv(out / f'{method}_windows.csv')
        episodes['block_start'] = ((episodes.end_interaction - 1) // 100000) * 100000
        grouped = episodes.groupby(['seed', 'block_start']).raw_return.agg(['mean', 'count']).reset_index()
        for row in grouped.to_dict('records'):
            blocks.append(dict(method=method, **row))
        record = dict(verified_transitions=sum(s['verified_transitions'] for s in seeds),
            verified_log_segments=sum(s['verified_segments'] for s in seeds),
            completed_episodes=sum(s['ended_episodes'] for s in seeds),
            truncated_episodes=sum(s['truncated_episodes'] for s in seeds),
            first100_episode_mean_across_seeds=float(np.mean([s['first100_mean'] for s in seeds])),
            last100_episode_mean_across_seeds=float(np.mean([s['last100_mean'] for s in seeds])),
            improved_first_vs_last100_seeds=sum(s['last100_mean'] > s['first100_mean'] for s in seeds))
        for name, frame in [('first100k', windows[windows.end <= 100000]),
                            ('last100k', windows[windows.end > 900000])]:
            record[name + '_raw_reward_per_1000_decisions'] = float(1000 * frame.raw_reward.sum() / frame.decisions.sum())
        summaries[method] = record
    save_csv(out / 'training_blocks_100k.csv', blocks)
    (out / 'training_comparison.json').write_text(json.dumps(summaries, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--historical', type=Path, required=True)
    parser.add_argument('--anova', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--reuse-historical', action='store_true', help='Reuse this audit outputs after verifying their manifest identity')
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    if args.reuse_historical:
        hm = read(args.historical / 'campaign.json')
        assert json.loads((args.output / 'historical_manifest_hash.json').read_text())['campaign_hash'] == digest(hm)
    else:
        hm = inspect_campaign(args.historical, 'historical', args.output, True)
    am = inspect_campaign(args.anova, 'anova', args.output, True)
    assert hm['runtime'] == am['runtime']
    for h, a in zip(hm['cells'], am['cells']):
        assert h['seed'] == a['seed'] and h['learning'] == a['learning'] and h['environment'] == a['environment']
        assert h['estimator'] == {k: v for k, v in a['estimator'].items() if k != 'split_test'}
        assert a['estimator']['split_test'] == 'anova'
    changed = [p for p in sorted(set(hm['source_hashes']) | set(am['source_hashes'])) if hm['source_hashes'].get(p) != am['source_hashes'].get(p)]
    (args.output / 'provenance.json').write_text(json.dumps(dict(historical_root=str(args.historical.resolve()),
        anova_root=str(args.anova.resolve()), historical_campaign_hash=digest(hm), anova_campaign_hash=digest(am),
        runtime_equal=True, cell_learning_environment_equal=True, estimator_change='split_test=anova',
        changed_source_files=changed, note='Source hashes differ: descriptive aligned-seed comparison, not strict paired exporter validation.'), indent=2))
    summarize_training(args.output)
    plots(args.output)


if __name__ == '__main__':
    main()
