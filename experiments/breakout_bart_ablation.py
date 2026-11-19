"""Paired Breakout OCAtari pilot for BART initialization, cadence and warm-up."""

import argparse
import copy
import json
from pathlib import Path
from statistics import mean

import numpy as np

from bayesian_rrtl.online.learners import PredictionCache
from bayesian_rrtl.online.multigame_protocol import cell_protocol, raw_protocol
from bayesian_rrtl.online.protocol import authorize, freeze, load, select_cell
from bayesian_rrtl.online.storage import binding, load_checkpoint, stream_rows
from experiments.compare_pong_progressive import (
    StopRequest, checkpoint_at, default_factory, evaluation_identity, train_cell, validate_status,
)
from experiments.pong_analysis_runtime import checked_read, checked_write, episode


VARIANTS = {
    'control': {},
    'warm_start': {'estimator': {'warm_start': True}},
    'interval_1024': {'learning': {'fit_interval': 1024}},
    'warmup_4096': {'learning': {'warmup_interactions': 4096}},
}


def protocols(config, historical):
    variants = config['variants']
    if (variants.get('control') != {} or len(variants) < 2
            or any(name not in VARIANTS or change != VARIANTS[name] for name, change in variants.items())):
        raise ValueError('Expected control and predeclared single-factor contrasts')
    raw = raw_protocol(historical)
    if (set(raw['games']) != {'Breakout'} or set(raw['learners']) != {'bart'}
            or raw['learning']['fit_interval'] != 512
            or raw['learning']['warmup_interactions'] != 512
            or any(raw['learners']['bart'][name] != value
                   for name, value in {'kernel': 'grow_prune', 'm': 20, 'burn_in': 50, 'draws': 50}.items())):
        raise ValueError('Unexpected historical BART control')
    raw.update(level='baseline', bart_review=None)
    raw['learners']['bart'].update(warm_start=False, first_fit_at_warmup=True)
    game = raw['games']['Breakout']
    checkpoints = config['checkpoints']
    game.update(training_seeds=config['training_seeds'], max_decisions=checkpoints[-1],
                milestones=[checkpoints[-1]], checkpoint_interval=checkpoints[0],
                auc_interval=checkpoints[:2], contrasts=[])
    game['evaluation']['intermediate'].update(checkpoints=checkpoints[:-1], seeds=config['intermediate_seeds'])
    game['evaluation']['final_development'].update(checkpoints=checkpoints[-1:], seeds=config['development_seeds'])
    game['budgets'].update(training_seconds_per_cell=config['training_seconds'],
                           evaluation_seconds_per_episode=config['evaluation_seconds'], fit_margin_seconds=10)
    raw['resources'] = {'cpus_per_cell': 1, 'memory_gib_per_cell': 8}
    raw['cells'] = [{'game': 'Breakout', 'pipeline': 'ocatari', 'learner': 'bart', 'seed': seed}
                    for seed in config['training_seeds']]
    result = {}
    for variant, change in variants.items():
        run = copy.deepcopy(raw)
        run['name'] = config.get('name', 'breakout_bart_ablation_20261110') + '/' + variant
        run['learning'].update(change.get('learning', {}))
        run['learners']['bart'].update(change.get('estimator', {}))
        result[variant] = run
    return result


def prepare(output, config, historical):
    for variant, raw in protocols(config, historical).items():
        path = output / variant
        freeze(raw, path)
        authorize(path, config['checkpoints'][-1],
                  'Paired development campaign with predeclared single-factor contrasts.', game='Breakout')


def smoke(output, config, historical):
    for variant, raw in protocols(config, historical).items():
        path = output / variant
        seed = config['training_seeds'][0]
        game = raw['games']['Breakout']
        game['environment']['horizon'] = 128
        game.update(training_seeds=[seed], max_decisions=96, milestones=[96],
                    checkpoint_interval=32, auc_interval=[32, 64])
        game['evaluation']['intermediate'].update(checkpoints=[32, 64], horizon=128, seeds=config['intermediate_seeds'][:1])
        game['evaluation']['final_development'].update(checkpoints=[96], horizon=128, seeds=config['development_seeds'][:1])
        game['budgets'].update(training_seconds_per_cell=300, evaluation_seconds_per_episode=120)
        raw['cells'] = raw['cells'][:1]
        raw['learning'].update(replay_capacity=128, batch_size_per_action=8)
        for key in ('fit_interval', 'warmup_interactions'):
            raw['learning'][key] //= 64
        raw['learners']['bart'].update(m=2, burn_in=2, draws=4)
        freeze(raw, path)
        authorize(path, 96, 'Technical integration on Breakout ALE, reduced synthetic budget.', game='Breakout')
        run_cell(output, variant, 0, 290)


class ServiceController:
    """Frozen greedy BART and time to first FIRE after each service opportunity."""

    def __init__(self, agent):
        self.model = agent.model
        self.encoder = agent.encoder
        self.cache = PredictionCache()
        self.fire = agent.config.actions.index('FIRE')
        self.records = []
        self.pending = ('reset', 0)

    def action(self, state, env, rng):
        q = self.cache.predict(self.model, self.encoder.transform([state]))[0]
        return int(rng.choice(np.flatnonzero(q == q.max())))

    def transition(self, before, action, after, reward, raw, terminal, truncated, env):
        ended = terminal or truncated
        lost = env.last_auxiliary['life_lost']
        if self.pending is not None and (action == self.fire or ended or lost):
            kind, start = self.pending
            self.records.append({'kind': kind, 'start': start, 'decisions': env.steps-start,
                                 'fired': action == self.fire})
            self.pending = None
        if lost and not ended:
            self.pending = ('life_lost', env.steps)


def evaluate(path, manifest, cell, status, stop):
    folder = path / cell['id']
    spec = cell_protocol(manifest, cell)
    result = {}
    for group in ('intermediate', 'final_development'):
        schedule = spec['evaluation'][group]
        factory = default_factory(cell).with_horizon(schedule['horizon'])
        for point in schedule['checkpoints']:
            ref = checkpoint_at(folder, status, point)
            payload, _ = load_checkpoint(folder, ref, binding(manifest, cell))
            agent = payload['session']['collector']['agent']
            result[point] = []
            for seed in schedule['seeds']:
                identity = {**evaluation_identity(manifest, cell, ref, group, seed), 'service': 'first_fire_v1'}
                target = folder / 'evaluations' / group / str(point) / str(seed)
                row = episode(factory, ServiceController(agent), target, identity, seed=seed, rng_seed=seed,
                              horizon=schedule['horizon'], seconds=spec['budgets']['evaluation_seconds_per_episode'],
                              stop=stop, checkpoint_every=512, pong_points=False)
                if row['status'] != 'completed':
                    raise RuntimeError(f'Partial development evaluation: {target}')
                result[point].append(row)
    return result


def summarize_cell(folder, status, spec, cell, evaluation):
    learning = cell['learning']
    fit_steps = []
    total = reused = 0
    initial_rows = {}
    for row in stream_rows(folder, status['log_head']):
        if row['fit'] is None:
            continue
        if not fit_steps:
            initial_rows = {a: v.get('samples', 0) for a, v in row['fit']['actions'].items()}
        fit_steps.append(row['interaction'])
        if len(fit_steps) > 1:
            for action in row['fit']['actions'].values():
                reused += action.get('reused_trees', 0)
                total += action.get('m', 0)
    expected = list(range(learning['warmup_interactions'], spec['max_decisions']+1, learning['fit_interval']))
    if fit_steps != expected:
        raise ValueError('Actual training cadence differs from the declared contrast')
    return {'training_seed': cell['seed'], 'checkpoint': status['checkpoint'], 'fits': len(fit_steps),
            'first_fit': fit_steps[0], 'first_fit_rows_by_action': initial_rows,
            'trees_reused_after_first_fit': reused, 'trees_fit_after_first_fit': total,
            'training_phase_seconds': status['metrics']['phase_seconds'],
            'development': {str(point): episodes for point, episodes in evaluation.items()}}


def run_cell(output, variant, index, attempt_seconds):
    path = output / variant
    manifest = load(path)
    cell = select_cell(manifest, str(index))
    with StopRequest() as stop:
        status = train_cell(path, str(index), attempt_seconds=attempt_seconds, stop=stop)
        validate_status(status, manifest, cell)
        if status['state'] != 'completed':
            raise RuntimeError(f'Partial training at {status["counter"]}: {status["reason"]}')
        results = evaluate(path, manifest, cell, status, stop)
    folder = path / cell['id']
    summary = summarize_cell(folder, status, cell_protocol(manifest, cell), cell, results)
    checked_write(folder / 'ablation_summary.json', summary)


def service_summary(episodes, kind):
    opportunities = [r for episode_row in episodes for r in episode_row['transitions'] if r['kind'] == kind]
    times = [r['decisions'] for r in opportunities if r['fired']]
    return {'opportunities': len(opportunities), 'fires': len(times),
            'fired_within_32': sum(n <= 32 for n in times),
            'censored_before_32': sum(not r['fired'] and r['decisions'] < 32 for r in opportunities),
            'mean_delay_given_fire': mean(times) if times else None}


def report(output, config):
    point = str(config['checkpoints'][-1])
    rows = {}
    for variant in config['variants']:
        rows[variant] = {}
        for seed in config['training_seeds']:
            path = output / variant / 'Breakout' / 'ocatari' / 'bart' / str(seed) / 'ablation_summary.json'
            cell = checked_read(path)
            episodes = cell['development'][point]
            if [row['seed'] for row in episodes] != config['development_seeds']:
                raise ValueError('Unpaired development episodes')
            rows[variant][str(seed)] = {
                'mean_return': mean(row['raw_return'] for row in episodes),
                'truncations': sum(row['truncated'] for row in episodes),
                'reset': service_summary(episodes, 'reset'), 'life_lost': service_summary(episodes, 'life_lost'),
                'training_phase_seconds': cell['training_phase_seconds'], 'fits': cell['fits'],
                'first_fit': cell['first_fit'], 'reused_trees': cell['trees_reused_after_first_fit'],
                'eligible_tree_updates': cell['trees_fit_after_first_fit'],
            }
    paired = {variant: {seed: row['mean_return'] - rows['control'][seed]['mean_return']
                       for seed, row in by_seed.items()} for variant, by_seed in rows.items() if variant != 'control'}
    checked_write(output / 'report.json', {'decision_budget': int(point), 'per_seed': rows,
                                           'paired_return_differences': paired, 'reserved_test_episodes': 0})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'smoke', 'fixed', 'run', 'report'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--historical', type=Path)
    parser.add_argument('--variant', choices=tuple(VARIANTS))
    parser.add_argument('--index', type=int)
    parser.add_argument('--task', type=int, help='Array index alternating variants within each seed')
    parser.add_argument('--attempt-seconds', type=int, default=14280)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.stage in ('prepare', 'smoke'):
        stage = prepare if args.stage == 'prepare' else smoke
        stage(args.output, config, checked_read(args.historical / 'campaign.json'))
    elif args.stage == 'fixed':
        from experiments.breakout_bart_fixed_start import audit
        audit(args.output, args.historical, config)
    elif args.stage == 'run':
        if args.task is None:
            if args.variant not in config['variants'] or args.index is None:
                parser.error('Choose a declared --variant and --index or pass --task')
            variant, index = args.variant, args.index
        else:
            if args.variant is not None or args.index is not None:
                parser.error('--task cannot be combined with --variant or --index')
            variants = tuple(config['variants'])
            if not 0 <= args.task < len(variants)*len(config['training_seeds']):
                parser.error('Array task outside the declared campaign')
            index, offset = divmod(args.task, len(variants))
            variant = variants[offset]
        run_cell(args.output, variant, index, args.attempt_seconds)
    else:
        report(args.output, config)


if __name__ == '__main__':
    main()
