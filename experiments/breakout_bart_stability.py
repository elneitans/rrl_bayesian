"""Separate, paired BART trials for inherited structure and frozen target cadence."""

import argparse
import copy
import json
from pathlib import Path
from statistics import mean

from bayesian_rrtl.online.multigame_protocol import cell_protocol, raw_protocol
from bayesian_rrtl.online.protocol import authorize, freeze, load, select_cell
from bayesian_rrtl.online.storage import stream_rows
from experiments.breakout_bart_ablation import evaluate, service_summary, summarize_cell
from experiments.compare_pong_progressive import StopRequest, train_cell, validate_status
from experiments.pong_analysis_runtime import checked_read, checked_write


VARIANTS = {
    'bounded_structure': {'warm_start': True, 'max_structure_distance': 2},
    'slow_target': {'target_update_interval': 4},
}


def protocols(config, historical):
    if config['variants'] != VARIANTS:
        raise ValueError('Expected the two separate, predeclared stability variants')
    seeds, points = config['training_seeds'], config['checkpoints']
    if (not seeds or len(seeds) != len(set(seeds)) or len(seeds) * len(VARIANTS) > 10
            or len(points) < 3 or points != sorted(set(points))
            or any(type(p) is not int or p <= 0 for p in points) or points[-1] > 1_000_000):
        raise ValueError('At most ten trainings with a positive calendar up to one million decisions')
    raw = raw_protocol(historical)
    estimator = raw['learners']['bart']
    expected = {'kernel': 'grow_prune', 'm': 20, 'burn_in': 50, 'draws': 50,
                'warm_start': False, 'first_fit_at_warmup': True}
    if (set(raw['games']) != {'Breakout'} or set(raw['learners']) != {'bart'}
            or raw['learning']['fit_interval'] != 1024 or raw['learning']['warmup_interactions'] != 512
            or any(estimator.get(k) != v for k, v in expected.items())
            or estimator.get('target_update_interval', 1) != 1
            or estimator.get('max_structure_distance') is not None):
        raise ValueError('Expected the frozen 1,024-decision BART reference protocol')
    raw.update(level='baseline', bart_review=None)
    game = raw['games']['Breakout']
    game.update(training_seeds=seeds, max_decisions=points[-1], milestones=[points[-1]],
                checkpoint_interval=points[0], auc_interval=points[:2], contrasts=[])
    game['evaluation']['intermediate'].update(checkpoints=points[:-1], seeds=config['intermediate_seeds'])
    game['evaluation']['final_development'].update(checkpoints=points[-1:], seeds=config['development_seeds'])
    game['budgets'].update(training_seconds_per_cell=config['training_seconds'],
                           evaluation_seconds_per_episode=config['evaluation_seconds'], fit_margin_seconds=10)
    raw['resources'] = {'cpus_per_cell': 1, 'memory_gib_per_cell': 8}
    raw['cells'] = [{'game': 'Breakout', 'pipeline': 'ocatari', 'learner': 'bart', 'seed': seed}
                    for seed in seeds]
    result = {}
    for name, changes in VARIANTS.items():
        variant = copy.deepcopy(raw)
        variant['name'] = config['name'] + '/' + name
        variant['learners']['bart'].update(changes)
        result[name] = variant
    return result


def prepare(output, config, historical):
    for variant, raw in protocols(config, historical).items():
        path = output / variant
        freeze(raw, path)
        authorize(path, config['checkpoints'][-1], 'User requested both separate BART improvements.', game='Breakout')


def verify_fits(folder, status, cell):
    estimator = cell['estimator']
    interval = estimator.get('target_update_interval', 1)
    limit = estimator.get('max_structure_distance')
    fits, repaired, rejected, skipped = 0, 0, 0, 0
    targets = []
    for row in stream_rows(folder, status['log_head']):
        if row['fit'] is None:
            continue
        fits += 1
        target = ((fits - 1) // interval) * interval
        if row['fit_id'] != fits or row['fit']['target_model_id'] != target:
            raise ValueError('Actual frozen-target cadence differs from the declared variant')
        targets.append(target)
        if limit is None or fits == 1:
            continue
        for action in row['fit']['actions'].values():
            if not action['updated']:
                if (action['reason'] != 'support_repair_exceeds_limit'
                        or action['repair_distance'] <= limit or action['structure_limit'] != limit):
                    raise ValueError('Unexpected skipped action in bounded-structure trial')
                skipped += 1
                continue
            distances = action['max_structure_distance_per_tree']
            repairs = action['support_repairs_per_tree']
            if (action['structure_limit'] != limit or len(distances) != action['m']
                    or len(repairs) != action['m'] or action['reset_trees'] != 0
                    or action['reused_trees'] != action['m']
                    or any(not 0 <= r <= d <= limit for r, d in zip(repairs, distances))):
                raise ValueError('A fitted tree exceeded its declared structural neighbourhood')
            repaired += sum(repairs)
            rejected += action['distance_rejections']
    return {'verified_fits': fits, 'target_update_interval': interval, 'target_model_ids': targets,
            'support_repair_nodes': repaired, 'distance_rejections': rejected,
            'actions_retained_for_support': skipped}


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
    summary['stability'] = verify_fits(folder, status, cell)
    checked_write(folder / 'ablation_summary.json', summary)


def smoke(output, config, historical):
    for variant, raw in protocols(config, historical).items():
        path = output / variant
        raw['cells'] = raw['cells'][:1]
        game = raw['games']['Breakout']
        game['environment']['horizon'] = 128
        game.update(training_seeds=config['training_seeds'][:1], max_decisions=96, milestones=[96],
                    checkpoint_interval=32, auc_interval=[32, 64])
        game['evaluation']['intermediate'].update(checkpoints=[32, 64], horizon=128,
                                                seeds=config['intermediate_seeds'][:1])
        game['evaluation']['final_development'].update(checkpoints=[96], horizon=128,
                                                     seeds=config['development_seeds'][:1])
        game['budgets'].update(training_seconds_per_cell=300, evaluation_seconds_per_episode=120)
        raw['learning'].update(fit_interval=16, warmup_interactions=8, replay_capacity=128,
                               batch_size_per_action=8)
        raw['learners']['bart'].update(m=2, burn_in=2, draws=4)
        freeze(raw, path)
        authorize(path, 96, 'Technical integration, not an additional performance trial.', game='Breakout')
        run_cell(output, variant, 0, 290)


def report(output, config):
    by_variant = {}
    final = str(config['checkpoints'][-1])
    for variant in VARIANTS:
        manifest = load(output / variant)
        by_variant[variant] = {}
        for cell in manifest['cells']:
            folder = output / variant / cell['id']
            summary = checked_read(folder / 'ablation_summary.json')
            status = checked_read(folder / 'status.json')
            if status['state'] != 'completed' or status['counter'] != config['checkpoints'][-1]:
                raise ValueError('Campaign contains an incomplete training')
            integrity = verify_fits(folder, status, cell)
            if integrity != summary['stability']:
                raise ValueError('Stability summary differs from its immutable training logs')
            for point, episodes in summary['development'].items():
                group = 'final_development' if point == final else 'intermediate'
                expected = config['development_seeds'] if point == final else config['intermediate_seeds']
                if [r['seed'] for r in episodes] != expected:
                    raise ValueError('Unexpected development evaluation seeds')
                for row in episodes:
                    state = checked_read(folder / 'evaluations' / group / point / str(row['seed']) / 'state.json')
                    if state['status'] != 'completed' or state['result'] != row:
                        raise ValueError('Evaluation differs from its individual result')
            episodes = summary['development'][final]
            by_variant[variant][str(cell['seed'])] = {
                'by_checkpoint': {p: mean(r['raw_return'] for r in rows)
                                  for p, rows in summary['development'].items()},
                'final_mean_return': mean(r['raw_return'] for r in episodes),
                'reset': service_summary(episodes, 'reset'), 'life_lost': service_summary(episodes, 'life_lost'),
                'truncations': sum(r['truncated'] for r in episodes), 'stability': integrity,
                'training_phase_seconds': summary['training_phase_seconds'],
            }
    checked_write(output / 'report.json', {'decision_budget': int(final), 'per_seed': by_variant,
        'paired_final_differences_structure_minus_target': {
            str(seed): by_variant['bounded_structure'][str(seed)]['final_mean_return']
                       - by_variant['slow_target'][str(seed)]['final_mean_return']
            for seed in config['training_seeds']}, 'reserved_test_episodes': 0})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'smoke', 'run', 'report'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--historical', type=Path)
    parser.add_argument('--task', type=int)
    parser.add_argument('--attempt-seconds', type=int, default=85000)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.stage in ('prepare', 'smoke'):
        historical = checked_read(args.historical / 'campaign.json')
        (prepare if args.stage == 'prepare' else smoke)(args.output, config, historical)
    elif args.stage == 'run':
        if args.task is None or not 0 <= args.task < len(VARIANTS) * len(config['training_seeds']):
            parser.error('Choose an array task within the ten declared trials')
        index, offset = divmod(args.task, len(VARIANTS))
        run_cell(args.output, tuple(VARIANTS)[offset], index, args.attempt_seconds)
    else:
        report(args.output, config)


if __name__ == '__main__':
    main()
