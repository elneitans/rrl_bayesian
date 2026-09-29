"""Offline fixed-transition FQI: BART posterior mean, CART and nominal ExtraTrees."""
import argparse
from dataclasses import dataclass
from importlib.metadata import version
import os
from pathlib import Path
import pickle
import time

import numpy as np

from bayesian_rrtl.agent import calibration_design
from bayesian_rrtl.atari import AtariConfig
from bayesian_rrtl.controls import fit_cart
from bayesian_rrtl.ensemble import BARTRegressor
from bayesian_rrtl.evaluation import paired_bootstrap
from bayesian_rrtl.priors import calibrate_noise
from bayesian_rrtl.targets import frozen_q_targets
from experiments.pong_analysis_common import (ROOT, encoded, load_manifest, load_npz,
    manifest_encoder, read, sha, source_path, transition_from_arrays, write_new)

EXTRA_TREES_VERSION = '1.9.1'
METHODS = ('bart', 'cart', 'extra_trees')


def nominal_indicators(batch):
    """Each category and missingness gets its own column; no ordinal threshold."""
    return np.column_stack([(batch.values[:, j] == level).astype(np.float32)
        for j, key in enumerate(batch.catalog)
        for level in (-1, *(range(3) if key.type == 'comparative' else range(2)))])


@dataclass
class FittedAction:
    method: str
    estimator: object
    catalog_hash: str

    def predict(self, batch):
        if batch.catalog_hash != self.catalog_hash:
            raise ValueError('FQI catalog mismatch')
        if self.method == 'bart':
            return self.estimator.predict_mean(batch)
        if self.method == 'cart':
            return self.estimator.predict(batch)
        return self.estimator.predict(nominal_indicators(batch))


@dataclass
class FittedQ:
    actions: tuple[str, ...]
    estimators: tuple[FittedAction | None, ...]
    catalog_hash: str

    def predict(self, batch):
        if batch.catalog_hash != self.catalog_hash:
            raise ValueError('FQI catalog mismatch')
        return np.column_stack([np.zeros(len(batch)) if model is None else model.predict(batch)
                                for model in self.estimators])


def fit_action(method, batch, y, config_data, seed, *, smoke=False):
    from bayesian_rrtl.q_config import BayesianQConfig
    config = BayesianQConfig(**config_data)
    start = time.perf_counter()
    if method == 'bart':
        calibration = calibrate_noise(config.scale.normalize(y), calibration_design(batch),
            nu=config.nu, quantile=config.noise_quantile, variance_floor=config.variance_floor)
        model = BARTRegressor(m=2 if smoke else config.m, tree_prior=config.tree_prior,
            noise_prior=calibration.prior, scale=config.scale, k=config.k, kernel=config.kernel).fit(
                batch, y, np.random.default_rng(seed), burn_in=2 if smoke else config.burn_in,
                draws=3 if smoke else config.draws)
        complexity = {'trees': model.m, 'leaves': model.trace[-1].total_leaves,
                      'calibration_method': calibration.method}
    elif method == 'cart':
        model = fit_cart(batch, y, config.tree_prior)
        complexity = {'leaves': sum(node.is_leaf for _, node, _ in model.walk(batch)),
                      'max_depth': max(len(path) for path, _, _ in model.walk(batch))}
    elif method == 'extra_trees':
        if version('scikit-learn') != EXTRA_TREES_VERSION:
            raise ValueError(f'ExtraTrees requires scikit-learn {EXTRA_TREES_VERSION}')
        from sklearn.ensemble import ExtraTreesRegressor
        model = ExtraTreesRegressor(n_estimators=20, max_depth=4, min_samples_leaf=2,
            random_state=seed, n_jobs=1).fit(nominal_indicators(batch), y)
        complexity = {'trees': 20, 'leaves': sum(tree.get_n_leaves() for tree in model.estimators_)}
    else:
        raise ValueError('Unknown control')
    return FittedAction(method, model, batch.catalog_hash), {'seconds': time.perf_counter()-start,
        'samples': len(y), 'complexity': complexity, 'random_state': seed}


def _data(analysis, pipeline, kind):
    folder = Path(analysis) / pipeline / kind
    meta = read(folder / 'metadata.json')
    return meta, load_npz(folder / f'{kind}.npz', meta['npz_sha256'])


def regression_benchmark(analysis, pipeline, *, smoke=False):
    meta, data = _data(analysis, pipeline, 'fixed_fit')
    encoder = manifest_encoder(meta)
    results = {}
    for action, action_meta in enumerate(meta['actions']):
        prefix = f'a{action}_'
        train = data[prefix+'regression_train']
        batch = encoded(data[prefix+'values'], data[prefix+'observed'], encoder)
        targets = np.asarray(data[prefix+'targets'], dtype=float)
        if not train.any() or np.all(train):
            results[action_meta['action']] = {'status': 'no_heldout_episode',
                'data_hash': action_meta['data_hash']}
            continue
        models = {}
        for method in METHODS:
            model, details = fit_action(method, batch.take(np.flatnonzero(train)), targets[train],
                meta['config'], 570001 + action, smoke=smoke)
            predicted = model.predict(batch.take(np.flatnonzero(~train)))
            models[method] = {**details, 'mse_heldout_episode': float(np.mean((predicted-targets[~train])**2)),
                'heldout_samples': int((~train).sum())}
        results[action_meta['action']] = {'status': 'checked', 'models': models,
            'data_hash': action_meta['data_hash'],
            'train_target_sha256': __import__('hashlib').sha256(targets[train].tobytes()).hexdigest(),
            'heldout_target_sha256': __import__('hashlib').sha256(targets[~train].tobytes()).hexdigest()}
    return {'pipeline': pipeline, 'kind': 'same_fixed_X_y_regression', 'actions': results,
        'interpretation': 'Held-out error uses whole pilot episodes excluded from each fit.'}


def selected_transitions(analysis, pipeline, dataset='pilot'):
    kind = 'fixed_transitions' if dataset == 'pilot' else 'assisted_fixed_transitions'
    folder = Path(analysis) / pipeline / kind
    meta = read(folder / 'metadata.json')
    data = load_npz(folder / 'fixed_transitions.npz', meta['npz_sha256'])
    encoder = manifest_encoder(meta)
    selected = np.flatnonzero(data['fqi_selected'] & data['episode_train'])
    transitions = tuple(transition_from_arrays(data, int(i), encoder) for i in selected)
    if len(transitions) != meta['fqi_selected_count']:
        raise ValueError('Frozen transition selection mismatch')
    return meta, encoder, transitions, data


def fit_epoch(method, transitions, encoder, config_data, previous, epoch, *, smoke=False):
    from bayesian_rrtl.q_config import BayesianQConfig
    from bayesian_rrtl.replay import concatenate_states
    config = BayesianQConfig(**config_data)
    targets = frozen_q_targets(transitions, previous, encoder, gamma=config.gamma,
                               bootstrap_on_truncation=config.bootstrap_on_truncation)
    estimators = []
    details = []
    for action in range(len(config.actions)):
        indices = [i for i, t in enumerate(transitions) if t.action == action]
        if not indices:
            raise ValueError(f'No selected transitions for action {action}')
        batch = concatenate_states([transitions[i].state for i in indices], encoder)
        model, summary = fit_action(method, batch, targets[indices], config_data,
            580001 + epoch * 100 + action, smoke=smoke)
        estimators.append(model)
        details.append({'action': config.actions[action], **summary,
                        'target_min': float(targets[indices].min()),
                        'target_max': float(targets[indices].max())})
    # Only now publish Q_k. Earlier actions never enter y_k.
    return FittedQ(config.actions, tuple(estimators), encoder.catalog_hash), details


def evaluate_model(model, factory, seeds, horizon, seconds_per_episode):
    rows = []
    reference = None
    rngs = {seed: np.random.default_rng(seed + 590001) for seed in seeds}
    for seed in seeds:
        env = factory()
        started = time.monotonic()
        try:
            state = env.reset(seed=seed)
            for step in range(1, horizon + 1):
                if time.monotonic()-started >= seconds_per_episode:
                    rows.append({'seed': seed, 'status': 'partial', 'steps': step-1, 'reason': 'time_budget'})
                    break
                batch = model_encoder(model, env).transform([state])
                q = model.predict(batch)[0]
                if reference is None:
                    reference = (batch, q.copy())
                ties = np.flatnonzero(q == q.max())
                action = int(rngs[seed].choice(ties))
                state, _, raw, terminal, truncated = env.step(action)
                if step == 1:
                    total = won = lost = 0.
                    actions = {}
                total += raw
                won += max(raw, 0.)
                lost += max(-raw, 0.)
                actions[env.actions[action]] = actions.get(env.actions[action], 0) + 1
                if terminal or truncated:
                    break
            else:
                terminal, truncated = False, True
            if not rows or rows[-1].get('seed') != seed:
                rows.append({'seed': seed, 'status': 'completed', 'steps': step,
                    'terminated': bool(terminal), 'truncated': bool(truncated or not terminal),
                    'return_scope': 'complete_game' if terminal else 'horizon_limited',
                    'raw_return': total, 'points_won': won, 'points_lost': lost,
                    'actions': actions, 'seconds': time.monotonic()-started})
        finally:
            env.close()
    if reference is not None:
        np.testing.assert_array_equal(model.predict(reference[0])[0], reference[1])
    return rows


def model_encoder(model, env):
    encoder = env.encoder()
    if encoder.catalog_hash != model.catalog_hash:
        raise ValueError('Evaluation environment catalog mismatch')
    return encoder


def save_model(path, model):
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    return sha(path)


def run_fqi(analysis, pipeline, output, *, smoke=False, dataset='pilot'):
    manifest = load_manifest(analysis)
    specification = manifest['config']
    meta, encoder, transitions, _ = selected_transitions(analysis, pipeline, dataset)
    if not transitions:
        raise ValueError('No selected transitions')
    if not output.exists():
        output.mkdir(parents=True)
        write_new(output / 'identity.json', {'analysis_manifest_sha256': sha(Path(analysis)/'manifest.json'),
            'dataset_sha256': meta['npz_sha256'], 'pipeline': pipeline, 'smoke': smoke,
            'methods': METHODS, 'dataset': dataset,
            'epochs': [1] if smoke else specification['fqi_epochs']})
    elif read(output / 'identity.json')['dataset_sha256'] != meta['npz_sha256']:
        raise ValueError('FQI output belongs to another dataset')
    from bayesian_rrtl.comparison import PongEnvironmentFactory
    source_protocol = read(source_path(specification['source_campaign']) / 'protocol.json')['protocol']
    factory = PongEnvironmentFactory(AtariConfig(**source_protocol['environments'][pipeline]))
    horizon = specification['smoke_horizon'] if smoke else specification['horizon']
    factory = factory.with_horizon(horizon)
    seeds = [specification['smoke_seed']] if smoke else specification['fqi_evaluation_seeds']
    epochs = [1] if smoke else specification['fqi_epochs']
    for method in METHODS:
        path = output / method
        path.mkdir(exist_ok=True)
        previous = FittedQ(tuple(meta['config']['actions']), (None,)*len(meta['config']['actions']), encoder.catalog_hash)
        for epoch in range(1, max(epochs)+1):
            model_path = path / f'epoch_{epoch}.pkl'
            if epoch in epochs and model_path.exists():
                with model_path.open('rb') as handle:
                    previous = pickle.load(handle)  # trusted job-owned model
            else:
                current, details = fit_epoch(method, transitions, encoder, meta['config'], previous,
                                             epoch, smoke=smoke)
                previous = current
                if epoch in epochs:
                    save_model(model_path, current)
                    write_new(path / f'fit_{epoch}.json', {'epoch': epoch, 'method': method,
                        'details': details, 'model_sha256': sha(model_path),
                        'targets_recomputed_from': epoch-1})
            if epoch in epochs:
                evaluation = path / f'eval_{epoch}.json'
                if not evaluation.exists():
                    rows = evaluate_model(previous, factory, seeds, horizon,
                        specification['max_seconds_per_episode'])
                    write_new(evaluation, {'method': method, 'epoch': epoch, 'rows': rows,
                        'model_sha256': sha(model_path), 'dataset_sha256': meta['npz_sha256'],
                        'pipeline': pipeline})
                print(pipeline, method, epoch, 'evaluated', flush=True)
    return summarize_fqi(output, epochs)


def summarize_fqi(output, epochs):
    report = {'epochs': {}, 'methods': METHODS}
    for epoch in epochs:
        records = {m: read(output / m / f'eval_{epoch}.json')['rows'] for m in METHODS}
        completed = {m: {r['seed']: r['raw_return'] for r in rows if r['status']=='completed'}
                     for m, rows in records.items()}
        common = sorted(set.intersection(*(set(v) for v in completed.values())))
        report['epochs'][epoch] = {'results': records, 'complete_paired_seeds': common,
            'paired_vs_cart': {m: paired_bootstrap({s: completed[m][s] for s in common},
                {s: completed['cart'][s] for s in common}) for m in ('bart', 'extra_trees')}
                if common else {},
            'limitation': 'Truncated/time-budget episodes are identified separately; no complete-game claim.'}
    path = output / 'report.json'
    if path.exists():
        path.unlink()
    write_new(path, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', type=Path, required=True)
    parser.add_argument('--pipeline', choices=('ocatari','visual'), required=True)
    parser.add_argument('--stage', choices=('regression','fqi'), required=True)
    parser.add_argument('--dataset', choices=('pilot', 'assisted'), default='pilot')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    load_manifest(args.analysis)
    if args.dataset == 'assisted' and (args.pipeline != 'ocatari' or args.stage != 'fqi'):
        parser.error('Assisted transitions are only collected for OCAtari FQI')
    output = args.analysis / args.pipeline / (
        ('fqi_assisted_' if args.dataset == 'assisted' else 'fqi_') + 'smoke' if args.smoke
        else ('fqi_assisted' if args.dataset == 'assisted' else 'fqi'))
    if args.stage == 'regression':
        report = regression_benchmark(args.analysis, args.pipeline, smoke=args.smoke)
        output.mkdir(parents=True, exist_ok=True)
        write_new(output / 'regression.json', report)
    else:
        run_fqi(args.analysis, args.pipeline, output, smoke=args.smoke, dataset=args.dataset)


if __name__ == '__main__':
    main()
