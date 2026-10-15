"""B10: explicit v2 common Pong protocols, independent of historic four-cell v1."""
import copy
import json
from dataclasses import asdict
import math
import os
import platform
from pathlib import Path
import sys
from importlib.metadata import PackageNotFoundError, version

from bayesian_rrtl.games.pong import PONG_ACTIONS, PongRelationsV1
from bayesian_rrtl.online.learners import (CARTConfig, ExtraTreesConfig, LEARNERS, LearningConfig)
from bayesian_rrtl.baseline_adapter import BaselineLearnerConfig
from bayesian_rrtl.q_config import BayesianQConfig
from experiments.pong_analysis_runtime import checked_read, checked_write, digest, bytes_hash, unit_lock
from experiments.pong_common_contract import CommonPongConfig

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = 'pong_progressive_v2'


def source_hashes():
    paths = list((ROOT/'bayesian_rrtl').rglob('*.py')) + list((ROOT/'experiments').glob('*.py'))
    paths += [ROOT/'preprocessing.py', ROOT/'relational_regresion_tree.py']
    return {str(p.relative_to(ROOT)): bytes_hash(p.read_bytes()) for p in sorted(paths)}


def runtime(learner=None):
    names = ['numpy', 'scipy', 'gymnasium', 'ale-py', 'ocatari']
    if learner == 'extra_trees':
        names += ['scikit-learn']
    result = {'python': sys.version.split()[0], 'platform': sys.platform, 'machine': platform.machine(),
              'numeric_threads': {k: os.environ.get(k) for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                                                               'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')}}
    for name in names:
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result


def resolve(raw):
    if raw.get("schema") == "atari_comparison_v3":
        from .multigame_protocol import resolve as resolve_multigame
        return resolve_multigame(raw)
    p = copy.deepcopy(raw)
    required = {'schema', 'name', 'level', 'pipelines', 'rom_sha256', 'training_horizon',
                'learners', 'learning', 'exploration', 'training_seeds', 'evaluation',
                'max_decisions', 'milestones', 'checkpoint_interval', 'budgets',
                'cache_bytes', 'bart_review', 'primary_checkpoint'}
    if set(p) != required or p['schema'] != SCHEMA or p['level'] not in ('smoke', 'baseline', 'long'):
        raise ValueError('Invalid progressive protocol schema/fields/level')
    if not isinstance(p['name'], str) or not p['name'] or p['primary_checkpoint'] != 'final':
        raise ValueError('Protocol requires name and final primary checkpoint')
    if (not p['pipelines'] or len(set(p['pipelines'])) != len(p['pipelines'])
            or any(v not in ('visual', 'ocatari') for v in p['pipelines'])
            or not p['learners'] or set(p['learners'])-LEARNERS.keys()):
        raise ValueError('Unknown/empty pipeline or learner')
    for key in ('training_horizon', 'checkpoint_interval'):
        if type(p[key]) is not int or p[key] < 1:
            raise ValueError('Positive horizons/interval required')
    if type(p['cache_bytes']) is not int or p['cache_bytes'] < 0:
        raise ValueError('Invalid cache size')
    if set(p['exploration']) != {'epsilon_init', 'epsilon_min', 'epsilon_decay_steps'}:
        raise ValueError('Explicit exploration required')
    if set(p['learning']) != {'gamma', 'replay_capacity', 'batch_size_per_action',
                              'fit_interval', 'warmup_interactions', 'bootstrap_on_truncation'}:
        raise ValueError('Explicit shared learning configuration required')
    ev = p['evaluation']
    if set(ev) != {'intermediate', 'final_development', 'reserved_test'}:
        raise ValueError('Evaluation groups required')
    for group in ev.values():
        if set(group) != {'seeds', 'horizon', 'checkpoints'} or not group['seeds']:
            raise ValueError('Explicit evaluation seeds, horizon and checkpoints required')
        if type(group['horizon']) is not int or group['horizon'] < 1:
            raise ValueError('Invalid evaluation horizon')
        pts = group['checkpoints']
        if any(type(n) is not int or n < 0 for n in pts) or pts != sorted(set(pts)):
            raise ValueError('Invalid evaluation calendar')
    if ev['reserved_test']['checkpoints']:
        raise ValueError('Block B cannot schedule reserved test')
    seeds = p['training_seeds'] + sum((g['seeds'] for g in ev.values()), [])
    if not p['training_seeds'] or len(set(seeds)) != len(seeds) or any(
            type(n) is not int or not 0 <= n < 2**32 for n in seeds):
        raise ValueError('Seed groups must be disjoint unique uint32')
    if set(p['budgets']) != {'training_seconds_per_cell', 'evaluation_seconds_per_episode', 'fit_margin_seconds'}:
        raise ValueError('Separate per-unit budgets required')
    unresolved = []
    for key, value in p['budgets'].items():
        if value is None and p['level'] == 'long':
            unresolved.append('budgets.'+key)
        elif type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError('Invalid operational budget')
    maximum = p['max_decisions']
    if maximum is None and p['level'] == 'long':
        unresolved.append('max_decisions')
    elif type(maximum) is not int or maximum < 1:
        raise ValueError('Invalid maximum decisions')
    milestones = p['milestones']
    if (any(type(n) is not int or n < 1 for n in milestones) or milestones != sorted(set(milestones))
            or (maximum is not None and (not milestones or milestones[-1] != maximum))):
        raise ValueError('Milestones must end at frozen maximum')
    if maximum is not None and any(n > maximum for g in ev.values() for n in g['checkpoints']):
        raise ValueError('Evaluation beyond maximum')
    if maximum is not None and maximum not in ev['final_development']['checkpoints']:
        raise ValueError('Final development evaluation must include primary final checkpoint')
    if p['level'] == 'long' and 'bart' in p['learners'] and p['bart_review'] is None:
        unresolved.append('bart_review:A6/A7')
    if p['bart_review'] is not None:
        review = p['bart_review']
        if set(review) != {'conclusions', 'artifacts'} or not review['conclusions'] or not review['artifacts']:
            raise ValueError('BART review requires conclusions and diagnostic artifact hashes')
        for path, sha in review['artifacts'].items():
            if bytes_hash(Path(path).read_bytes()) != sha:
                raise ValueError('BART review artifact mismatch')
    if p['level'] == 'smoke' and not {'corrected_common', 'cart', 'bart'} <= p['learners'].keys():
        raise ValueError('Smoke requires all mandatory learners')
    cells = []
    for pipeline in p['pipelines']:
        env = asdict(CommonPongConfig(pipeline, p['rom_sha256'], p['training_horizon']))
        for learner in LEARNERS:
            if learner not in p['learners']:
                continue
            estimator = p['learners'][learner]
            if estimator is None:
                if p['level'] != 'long' or learner != 'bart':
                    raise ValueError('Unresolved estimator only allowed for long BART')
                unresolved.append('learners.bart')
            for seed in p['training_seeds']:
                learning = LearningConfig(actions=PONG_ACTIONS, seed=seed, **p['learning'], **p['exploration'],
                    validation_seeds=tuple(ev['intermediate']['seeds']+ev['final_development']['seeds']),
                    test_seeds=tuple(ev['reserved_test']['seeds']))
                if estimator is not None:
                    if learner == 'cart':
                        CARTConfig(**estimator)
                    elif learner == 'extra_trees':
                        ExtraTreesConfig(**estimator)
                    elif learner == 'bart':
                        BayesianQConfig(**asdict(learning), **estimator)
                    else:
                        if set(estimator) & {'seed', 'gamma', 'bootstrap_on_truncation'}:
                            raise ValueError('Duplicated baseline common fields')
                        BaselineLearnerConfig(**estimator, seed=seed, gamma=learning.gamma,
                            bootstrap_on_truncation=learning.bootstrap_on_truncation)
                cell = {'id': f'{pipeline}/{learner}/{seed}', 'pipeline': pipeline,
                        'learner': learner, 'seed': seed, 'environment': env,
                        'learning': asdict(learning), 'estimator': estimator}
                cells.append({**cell, 'hash': digest(cell)})
    pending = list(dict.fromkeys(unresolved))
    return {'schema': SCHEMA, 'protocol': p, 'cells': cells, 'unresolved': pending}


def freeze(raw, output):
    resolved = resolve(raw)  # validate before any output
    output = Path(output)
    with unit_lock(output):
        manifest = {**resolved, 'source_hashes': source_hashes(), 'runtime': runtime()}
        path = output/'campaign.json'
        if path.exists():
            if checked_read(path) != json.loads(json.dumps(manifest)):
                raise ValueError('Existing campaign differs; create a new variant')
        else:
            checked_write(path, manifest, replace=False)
        return load(output)


def load(output):
    manifest = checked_read(Path(output)/'campaign.json')
    from .multigame_protocol import raw_protocol
    resolved = resolve(raw_protocol(manifest))
    if digest(resolved) != digest({k: manifest[k] for k in resolved}):
        raise ValueError('Frozen scientific protocol mismatch')
    if manifest['source_hashes'] != source_hashes() or manifest['runtime'] != runtime():
        raise ValueError('Code/runtime changed; incompatible campaign resume')
    return manifest


def select_cell(manifest, selection):
    if str(selection).isdigit():
        index = int(selection)
        if not 0 <= index < len(manifest['cells']):
            raise ValueError('Invalid cell index')
        return manifest['cells'][index]
    for cell in manifest['cells']:
        if cell['id'] == selection:
            return cell
    raise ValueError('Unknown cell')


def authorize(output, stop_at, reason, measurements=None, game=None):
    manifest = load(output)
    p = manifest['protocol']
    if manifest['schema'] == 'atari_comparison_v3':
        if game not in p['games']:
            raise ValueError('Explicit declared game required for milestone authorization')
        p = {**p, **p['games'][game]}
    if manifest['unresolved'] or stop_at not in p['milestones'] or not reason.strip():
        raise ValueError('Unresolved protocol or invalid milestone/authorization reason')
    cost = None
    if manifest['protocol']['level'] == 'long':
        from .multigame_protocol import raw_protocol
        estimates = dry_run(raw_protocol(manifest), measurements)['estimates']
        if (sys.platform != 'linux' or len(estimates) != len(manifest['cells'])
                or any(e['evaluation_seconds_episode_observed_range'] is None for e in estimates)):
            raise ValueError('Long authorization requires compatible measured Linux costs for every cell')
        cost = {'measurements_sha256': digest(measurements), 'estimates': estimates}
    authorization = Path(output)/'authorization'/(game or '_all') if game else Path(output)/'authorization'
    with unit_lock(authorization):
        path = authorization/'ledger.json'
        ledger = checked_read(path) if path.exists() else []
        if ledger and stop_at < ledger[-1]['stop_at']:
            raise ValueError('Milestones cannot move backwards')
        if not ledger or stop_at != ledger[-1]['stop_at']:
            ledger.append({'stop_at': stop_at, 'reason': reason, 'campaign': digest(manifest), 'measured_cost_review': cost})
            checked_write(path, ledger)
        return stop_at


def authorized_stop(output, manifest, cell=None):
    if manifest['unresolved']:
        raise ValueError('Campaign is unresolved and inactive')
    from .multigame_protocol import cell_protocol
    p = cell_protocol(manifest, cell) if cell is not None else manifest['protocol']
    if p['level'] == 'smoke':
        return p['max_decisions']
    folder = Path(output)/'authorization'
    if manifest['schema'] == 'atari_comparison_v3':
        folder = folder/cell['game']
    path = folder/'ledger.json'
    if not path.exists():
        raise ValueError('Authorize a declared milestone before training')
    ledger = checked_read(path)
    if not ledger or any(r['campaign'] != digest(manifest) for r in ledger):
        raise ValueError('Invalid authorization identity')
    return ledger[-1]['stop_at']


def dry_run(raw, measurements=None):
    if raw.get("schema") == "atari_comparison_v3":
        from .multigame_protocol import dry_run as dry_multigame
        return dry_multigame(raw, measurements)
    resolved = resolve(raw)
    p = resolved['protocol']
    maximum = p['max_decisions']
    first_fit = max(1, p['learning']['warmup_interactions'], p['learning']['fit_interval'])
    fits = None if maximum is None else max(0, 1+(maximum-first_fit)//p['learning']['fit_interval'])
    checkpoint_count = None if maximum is None else 1 + maximum//p['checkpoint_interval'] + len(
        {n for n in p['milestones'] + [v for k,g in p['evaluation'].items() if k != 'reserved_test' for v in g['checkpoints']]
         if n and n % p['checkpoint_interval']})
    rows = [{'id': c['id'], 'seed': c['seed'], 'decisions': maximum,
             'expected_fits': fits if LEARNERS[c['learner']].batch else 0, 'expected_checkpoints': checkpoint_count,
             'evaluation_episodes': sum(len(g['seeds'])*len(g['checkpoints']) for key, g in p['evaluation'].items()
                                        if key != 'reserved_test')} for c in resolved['cells']]
    estimates = [v for c in resolved['cells']
                 if (v := estimate_cell_cost(c,p,measurements,maximum,checkpoint_count)) is not None]
    return {'resolved': resolved, 'workload': rows, 'estimates': estimates,
            'uncertainty': 'Missing compatible measurements means unknown time/storage, never zero.',
            'pending_authorizations': resolved['unresolved'] +
                ([] if p['level'] == 'smoke' else ['milestone authorization']) +
                (['compatible_Linux_cost_measurements', 'explicit_long_campaign_authorization'] if p['level'] == 'long' else []),
            'reserved_test_episodes_executed': 0}


def normalize_cost_environment(value):
    if value.get('game','Pong') == 'Pong':
        return {k:value[k] for k in ('pipeline','rom_sha256','horizon')}
    return value


def estimate_cell_cost(c, p, measurements, maximum, checkpoint_count):
    if not measurements or maximum is None:
        return None
    if 'game' in c['environment'] and c['environment']['game'] != 'Pong':
        from .environments import GameEnvironmentConfig
        env=GameEnvironmentConfig(**c['environment'])
        expected_contract=env.manifest()
        expected_catalog=env.relations().encoder().catalog_hash
    else:
        expected_contract=CommonPongConfig(c['environment']['pipeline'],c['environment']['rom_sha256'],c['environment']['horizon']).manifest()
        expected_catalog=PongRelationsV1().encoder().catalog_hash
    code=source_hashes()
    compatible = [r for r in measurements if digest(normalize_cost_environment(r['environment'])) == digest(normalize_cost_environment(c['environment']))
        and r.get('runtime_environment', {}).get('driver_provenance') == 'verified_atari'
        and r.get('runtime_environment', {}).get('contract') == expected_contract
        and r.get('catalog_hash') == expected_catalog
        and r.get('cache_bytes') == p['cache_bytes']
        and r.get('evaluation_horizons') == {k:g['horizon'] for k,g in p['evaluation'].items() if k != 'reserved_test'}
        and r['learner'] == c['learner'] and digest(r['learning_without_seed']) ==
        digest({k: v for k, v in c['learning'].items() if k != 'seed'})
        and r['estimator'] == c['estimator'] and r['runtime'] == runtime(c['learner'])
        and r['source_hashes'] == code]
    if compatible and maximum is not None:
        rates = [r['training_seconds']/r['decisions'] for r in compatible
                 if r['decisions'] and r.get('training_seconds') is not None]
        storage = [r['log_bytes']/r['decisions'] for r in compatible if r['decisions']]
        evaluation_rates = [seconds for r in compatible for seconds in r.get('evaluation_episode_seconds', [])]
        worker_rates = [r['evaluation_worker_seconds_per_completed_episode'] for r in compatible
                        if r.get('evaluation_worker_seconds_per_completed_episode') is not None]
        evaluation_count = sum(len(g['seeds'])*len(g['checkpoints']) for k,g in p['evaluation'].items() if k != 'reserved_test')
        peaks = [r['peak_rss_bytes'] for r in compatible if r.get('peak_rss_bytes') is not None]
        if rates:
            return {'cell': c['id'], 'training_seconds_observed_rate_range':
                [min(rates)*maximum, max(rates)*maximum], 'log_bytes_observed_rate_range':
                [min(storage)*maximum, max(storage)*maximum],
                'peak_rss_observed_range': [min(peaks), max(peaks)] if peaks else None,
                'incomplete_training_cost_records': sum(r.get('training_seconds') is None for r in compatible),
                'unknown_peak_rss_records': sum(r.get('peak_rss_bytes') is None for r in compatible),
                'checkpoint_bytes_before_retention_observed_size_projection':
                    [checkpoint_count*min(r['checkpoint_blob_bytes_range'][0] for r in compatible),
                     checkpoint_count*max(r['checkpoint_blob_bytes_range'][1] for r in compatible)],
                'evaluation_seconds_episode_observed_range': [min(evaluation_rates), max(evaluation_rates)] if evaluation_rates else None,
                'evaluation_seconds_observed_rate_projection':
                    [min(evaluation_rates)*evaluation_count, max(evaluation_rates)*evaluation_count] if evaluation_rates else None,
                'evaluation_worker_seconds_rate_projection':
                    [min(worker_rates)*evaluation_count, max(worker_rates)*evaluation_count] if worker_rates else None,
                'uncertainty': 'Observed rates only; incomplete attempt costs are unknown and excluded, never zero. Tree/replay growth and hardware can change cost. No maximum bound.'}
    return None
