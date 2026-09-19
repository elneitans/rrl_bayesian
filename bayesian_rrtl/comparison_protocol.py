"""Resolved, immutable four-cell Pong protocol. No emulator creation here."""
import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

from .atari import AtariConfig
from .baseline_adapter import BaselineLearnerConfig
from .comparison import ComparisonPolicyConfig, PongEnvironmentFactory
from .games.pong import PONG_ACTIONS
from .q_config import BayesianQConfig

ROOT = Path(__file__).resolve().parents[1]
CONTRASTS = ['visual/bart - visual/corrected_common', 'ocatari/bart - ocatari/corrected_common']


def digest(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sources():
    paths = sorted((ROOT / 'bayesian_rrtl').glob('*.py')) + sorted((ROOT / 'bayesian_rrtl/games').glob('*.py'))
    paths += [ROOT / name for name in ('preprocessing.py', 'relational_regresion_tree.py',
                                      'experiments/compare_pong.py')]
    return {str(path.relative_to(ROOT)): file_hash(path) for path in paths}


def resolve_protocol(raw):
    p = copy.deepcopy(raw)
    required = {'schema', 'purpose', 'environments', 'learners', 'training_seeds', 'validation_seeds',
                'test_seeds', 'interactions', 'curve_checkpoints', 'evaluation_episodes', 'evaluation_horizon',
                'contrasts', 'execution_order_seed', 'budgets', 'primary_checkpoint', 'checkpoint_interval'}
    if set(p) != required or p['schema'] != 'pong_comparison_v1' or p['purpose'] != 'development':
        raise ValueError('Invalid comparison protocol schema/fields/purpose')
    if (set(p['environments']) != {'visual', 'ocatari'} or set(p['learners']) != {'bart', 'corrected_common'}
            or p['contrasts'] != CONTRASTS or p['primary_checkpoint'] != 'final'):
        raise ValueError('Protocol must declare all four cells and within-pipeline contrasts')
    seeds = sum((p[k] for k in ('training_seeds', 'validation_seeds', 'test_seeds')), [])
    if (any(not p[k] for k in ('training_seeds', 'validation_seeds', 'test_seeds'))
            or any(type(s) is not int or not 0 <= s < 2**32 for s in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError('Training, validation and test seeds must be unique disjoint uint32 values')
    for name in ('interactions', 'evaluation_episodes', 'evaluation_horizon', 'checkpoint_interval'):
        if type(p[name]) is not int or p[name] < 1:
            raise ValueError('Invalid protocol budget')
    points = p['curve_checkpoints']
    if (any(type(n) is not int for n in points) or points != sorted(set(points))
            or len(points) < 2 or points[0] != 0 or points[-1] != p['interactions']):
        raise ValueError('Curve must include zero and final in increasing order')
    if p['evaluation_episodes'] != len(p['validation_seeds']):
        raise ValueError('Evaluation episodes must match all validation seeds')
    if type(p['execution_order_seed']) is not int or not 0 <= p['execution_order_seed'] < 2**32:
        raise ValueError('Invalid execution order seed')
    if set(p['budgets']) != {'training_seconds_per_run', 'evaluation_seconds_total'} or any(
            not np.isfinite(v) or v <= 0 for v in p['budgets'].values()):
        raise ValueError('Invalid operational budgets')
    policy = ComparisonPolicyConfig(validation_seeds=p['validation_seeds'], test_seeds=p['test_seeds'])
    cells = []
    for pipeline, expected_backend in (('visual', 'legacy_visual_pong_v1'), ('ocatari', 'ocatari_ram_v1')):
        env = PongEnvironmentFactory(AtariConfig(**p['environments'][pipeline])).config
        if env.extractor != expected_backend:
            raise ValueError('Pipeline backend mismatch')
        p['environments'][pipeline] = env.resolved_dict()
        for seed in p['training_seeds']:
            policy.validate_seed(seed)
            for learner in ('corrected_common', 'bart'):
                options = p['learners'][learner]
                if 'seed' in options:
                    raise ValueError('Training seeds are specified only by the protocol')
                if learner == 'corrected_common':
                    config = BaselineLearnerConfig(**options, seed=seed)
                else:
                    reserved = {'actions', 'validation_seeds', 'test_seeds', 'reward_min', 'reward_max'}
                    if reserved & options.keys():
                        raise ValueError('Actions, seeds and reward bounds derive from the environment/protocol')
                    config = BayesianQConfig(**options, seed=seed, actions=PONG_ACTIONS,
                        reward_min=env.reward_bounds[0], reward_max=env.reward_bounds[1],
                        validation_seeds=p['validation_seeds'], test_seeds=p['test_seeds'])
                    policy.validate_bart(config)
                if config.gamma != .99:
                    raise ValueError('Comparison requires gamma=.99')
                cell = {'id': f'{pipeline}/{learner}/{seed}', 'pipeline': pipeline, 'learner': learner,
                        'seed': seed, 'environment': env.resolved_dict(), 'config': asdict(config),
                        'policy': asdict(policy)}
                cells.append({**cell, 'hash': digest(cell)})
    horizons = [config['max_episode_steps'] for config in p['environments'].values()]
    if len(set(horizons)) != 1:
        raise ValueError('Both pipelines must declare the same training horizon')
    rng = np.random.default_rng(p['execution_order_seed'])
    order = []
    for seed in p['training_seeds']:
        block = [cell['id'] for cell in cells if cell['seed'] == seed]
        order.extend(str(item) for item in rng.permutation(block))
    return {'protocol': p, 'cells': cells, 'execution_order': order}


def audit_seeds(roots, candidates):
    """Scan existing JSON seed fields and CSV seed columns before freezing lists.

    Plans are intentions, not observations. Return file hashes and any evidence
    of use; do not inspect pickle or instantiate reserved-seed environments.
    """
    import csv
    candidates = set(candidates)
    hits, scanned = [], {}
    def inspect(value, path, key=''):
        if isinstance(value, dict):
            for k, v in value.items():
                inspect(v, path, k)
        elif isinstance(value, (list, tuple)):
            for v in value:
                inspect(v, path, key)
        elif 'seed' in key.lower():
            try:
                number = int(value)
            except (ValueError, TypeError):
                return
            if number in candidates:
                hits.append({'file': str(path), 'field': key, 'seed': number})
    for root in roots:
        for path in sorted(Path(root).rglob('*')):
            if path.suffix not in ('.json', '.csv') or not path.is_file():
                continue
            scanned[str(path)] = file_hash(path)
            try:
                if path.suffix == '.json':
                    inspect(json.loads(path.read_text()), path)
                else:
                    with path.open() as handle:
                        for row in csv.DictReader(handle):
                            inspect(row, path)
            except (ValueError, UnicodeError):
                raise ValueError(f'Could not audit seed-bearing artifact: {path}')
    return {'candidates': sorted(candidates), 'hits': hits, 'files_scanned': len(scanned),
            'inventory_sha256': digest(scanned), 'scope': 'JSON seed fields and CSV seed columns in supplied roots'}
