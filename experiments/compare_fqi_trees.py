"""Offline fixed-transition FQI: BART posterior mean, CART and nominal ExtraTrees."""
import argparse
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
import pickle
import time

import numpy as np

from bayesian_rrtl.agent import calibration_design
from bayesian_rrtl.atari import AtariConfig
from bayesian_rrtl.controls import fit_cart
from bayesian_rrtl.ensemble import BARTRegressor
from bayesian_rrtl.evaluation import paired_bootstrap
from bayesian_rrtl.online.learners import ExtraTreesConfig, nominal_indicators
from bayesian_rrtl.priors import calibrate_noise
from bayesian_rrtl.targets import frozen_q_targets
from experiments.pong_analysis_common import (encoded, load_manifest, load_npz,
    manifest_encoder, read, sha, transition_from_arrays, pilot_protocol, portable_metadata)
from experiments.pong_analysis_runtime import (WorkUnit, array_hash, atomic_bytes, atomic_json,
    bytes_hash, episode, extension_arguments, identity, mode_root, unit_lock)

EXTRA_TREES_VERSION = '1.9.1'
METHODS = ('bart', 'cart', 'extra_trees')




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
        if self.method == 'extra_trees':
            return self.estimator.predict(nominal_indicators(batch))
        raise ValueError('Unknown fitted estimator')


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


def fit_action(method, batch, y, config_data, seed, *, smoke=False, extra_trees_config=None):
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
        options = ExtraTreesConfig(**(extra_trees_config or {}))
        if version('scikit-learn') != options.sklearn_version:
            raise ValueError(f'ExtraTrees requires scikit-learn {EXTRA_TREES_VERSION}')
        from sklearn.ensemble import ExtraTreesRegressor
        model = ExtraTreesRegressor(n_estimators=options.n_estimators, max_depth=options.max_depth,
            min_samples_leaf=options.min_samples_leaf,
            random_state=seed, n_jobs=1).fit(nominal_indicators(batch), y)
        complexity = {'config': asdict(options), 'trees': options.n_estimators, 'leaves': int(sum(tree.get_n_leaves() for tree in model.estimators_))}
    else:
        raise ValueError('Unknown control')
    return FittedAction(method, model, batch.catalog_hash), {'seconds': time.perf_counter()-start,
        'samples': len(y), 'complexity': complexity, 'random_state': seed}


def _data(analysis, pipeline, kind, dataset_folder=None):
    folder = Path(dataset_folder) if dataset_folder else Path(analysis) / pipeline / kind
    meta = portable_metadata(folder)
    return meta, load_npz(folder / f'{kind}.npz', meta['npz_sha256'])


def regression_benchmark(analysis, pipeline, *, smoke=False, output=None, manifest=None, dataset_folder=None, extra_trees_config=None):
    meta, data = _data(analysis, pipeline, 'fixed_fit', dataset_folder)
    if meta.get('mode') == 'smoke' and not smoke:
        raise ValueError('Smoke dataset cannot satisfy a full regression')
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
            expected = (identity(manifest, pipeline=pipeline, stage='regression', method=method,
                dataset=meta.get('dataset_kind', 'common_fixed_fit' if 'common_protocol' in meta else 'pilot_fixed_fit'),
                variant=f'a{action}', smoke=smoke, seeds=[570001+action],
                effective={'learning': meta['config'], 'extra_trees': asdict(ExtraTreesConfig(**(extra_trees_config or {})))}
                    if method == 'extra_trees' else meta['config'], hashes={'data': meta['npz_sha256'], 'train_mask': array_hash(train), 'targets': array_hash(targets),
                                               'catalog': encoder.catalog_hash}) if manifest else None)
            models[method] = regression_unit(output/method/f'a{action}' if output else None,
                expected, method, batch, targets, train, meta['config'], 570001+action, smoke=smoke,
                extra_trees_config=extra_trees_config)
        results[action_meta['action']] = {'status': 'checked', 'models': models,
            'data_hash': action_meta['data_hash'],
            'train_target_sha256': __import__('hashlib').sha256(targets[train].tobytes()).hexdigest(),
            'heldout_target_sha256': __import__('hashlib').sha256(targets[~train].tobytes()).hexdigest()}
    return {'pipeline': pipeline, 'kind': 'same_fixed_X_y_regression', 'actions': results,
        'interpretation': meta.get('holdout_limitation', 'Whole holdout episodes excluded from all target-generator fits and calibration.')}



def regression_unit(folder, expected, method, batch, targets, train, config_data, seed, *, smoke, extra_trees_config=None):
    from contextlib import nullcontext
    with unit_lock(folder) if folder else nullcontext():
        unit = WorkUnit(folder, expected) if folder else None
        saved = unit.load() if unit else None
        if saved and saved['status'] == 'completed':
            return saved['result']
        model, details = fit_action(method, batch.take(np.flatnonzero(train)), targets[train],
                                    config_data, seed, smoke=smoke, extra_trees_config=extra_trees_config)
        predicted = model.predict(batch.take(np.flatnonzero(~train)))
        result = {**details, 'mse_heldout_episode': float(np.mean((predicted-targets[~train])**2)),
                  'heldout_samples': int((~train).sum())}
        if unit:
            unit.commit('completed', result=result, artifacts={'model': pickle.dumps(model)})
        return result


def selected_transitions(analysis, pipeline, dataset='pilot', dataset_folder=None):
    if dataset == 'common' and dataset_folder is None:
        raise ValueError('Common data require an explicit --dataset-folder')
    kind = 'fixed_transitions' if dataset == 'pilot' else 'assisted_fixed_transitions'
    folder = Path(dataset_folder) if dataset_folder else Path(analysis) / pipeline / kind
    meta = portable_metadata(folder)
    data = load_npz(folder / 'fixed_transitions.npz', meta['npz_sha256'])
    encoder = manifest_encoder(meta)
    selected = np.flatnonzero(data['fqi_selected'] & data['episode_train'])
    transitions = tuple(transition_from_arrays(data, int(i), encoder) for i in selected)
    if len(transitions) != meta['fqi_selected_count']:
        raise ValueError('Frozen transition selection mismatch')
    return meta, encoder, transitions, data


def fit_epoch(method, transitions, encoder, config_data, previous, epoch, *, smoke=False, extra_trees_config=None):
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
            580001 + epoch * 100 + action, smoke=smoke, extra_trees_config=extra_trees_config)
        estimators.append(model)
        details.append({'action': config.actions[action], **summary,
                        'target_sha256': array_hash(targets[indices]),
                        'target_min': float(targets[indices].min()),
                        'target_max': float(targets[indices].max())})
    # Only now publish Q_k. Earlier actions never enter y_k.
    return FittedQ(config.actions, tuple(estimators), encoder.catalog_hash), details


@dataclass
class GreedyController:
    model: FittedQ

    def action(self, state, env, rng):
        batch = model_encoder(self.model, env).transform([state])
        q = self.model.predict(batch)[0]
        return int(rng.choice(np.flatnonzero(q == q.max())))


def evaluate_model(model, factory, seeds, horizon, seconds_per_episode, *, output=None,
                   expected=None, extend_seconds=0., extension_id=None):
    import tempfile
    if output is None:
        with tempfile.TemporaryDirectory() as temporary:
            return evaluate_model(model, factory, seeds, horizon, seconds_per_episode,
                                  output=Path(temporary), expected={'model': bytes_hash(pickle.dumps(model))})
    model_hash = bytes_hash(pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL))
    rows = []
    for seed in seeds:
        unit_id = {**expected, 'episode_seed': seed, 'model_hash': model_hash, 'horizon': horizon}
        rows.append(episode(factory, GreedyController(model), output/str(seed), unit_id,
            seed=seed, horizon=horizon, seconds=seconds_per_episode, rng_seed=seed+590001,
            extend_seconds=extend_seconds, extension_id=extension_id))
    if bytes_hash(pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)) != model_hash:
        raise ValueError('Evaluation mutated frozen model')
    return rows


def model_encoder(model, env):
    encoder = env.encoder()
    if encoder.catalog_hash != model.catalog_hash:
        raise ValueError('Evaluation environment catalog mismatch')
    return encoder


def save_model(path, model):
    atomic_bytes(path, pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL))
    return sha(path)


def fit_iteration(folder, expected, method, transitions, encoder, config_data, previous, epoch, *, smoke=False, extra_trees_config=None):
    with unit_lock(folder):
        unit = WorkUnit(folder, expected)
        saved = unit.load()
        if saved and saved['status'] == 'completed':
            return unit.pickle(saved, 'model'), saved['result']
        current, details = fit_epoch(method, transitions, encoder, config_data, previous, epoch, smoke=smoke, extra_trees_config=extra_trees_config)
        raw = pickle.dumps(current, protocol=pickle.HIGHEST_PROTOCOL)
        result = {'epoch': epoch, 'method': method, 'details': details, 'model_sha256': bytes_hash(raw),
                  'target_hashes': [v['target_sha256'] for v in details],
                  'previous_model_sha256': expected['hashes']['previous_model']}
        unit.commit('completed', result=result, artifacts={'model': raw})
        return current, result


def run_fqi(analysis, pipeline, output, *, smoke=False, dataset='pilot',
            extend_seconds=0., extension_id=None, dataset_folder=None, extra_trees_config=None):
    manifest = load_manifest(analysis, writable=True)
    cfg = manifest['config']
    meta, encoder, transitions, _ = selected_transitions(analysis, pipeline, dataset, dataset_folder)
    dataset = meta.get('dataset_kind', dataset)
    if meta.get('mode') == 'smoke' and not smoke:
        raise ValueError('Smoke dataset cannot satisfy full FQI')
    if not transitions:
        raise ValueError('No selected transitions')
    from bayesian_rrtl.comparison import PongEnvironmentFactory
    protocol = pilot_protocol(manifest)
    horizon = cfg['smoke_horizon'] if smoke else cfg['horizon']
    if 'common_protocol' in meta:
        from experiments.pong_common_contract import CommonPongConfig, CommonPongFactory, validate_contract
        validate_contract(cfg['common_protocol'], meta['common_protocol'])
        factory = CommonPongFactory(CommonPongConfig(pipeline, meta['common_protocol']['rom_sha256'], horizon))
    else:
        factory = PongEnvironmentFactory(AtariConfig(**protocol['environments'][pipeline])).with_horizon(horizon)
    seeds = [cfg['smoke_seed']] if smoke else cfg['fqi_evaluation_seeds']
    epochs = [1] if smoke else cfg['fqi_epochs']
    # Recover and fit all iterations before evaluating. An interrupted evaluation
    # cannot hide an already committed model or force it to be fitted again.
    for method in METHODS:
        previous = FittedQ(tuple(meta['config']['actions']), (None,)*len(meta['config']['actions']), encoder.catalog_hash)
        previous_hash = bytes_hash(pickle.dumps(previous, protocol=pickle.HIGHEST_PROTOCOL))
        for epoch in range(1, max(epochs)+1):
            target = frozen_q_targets(transitions, previous, encoder, gamma=meta['config']['gamma'],
                                     bootstrap_on_truncation=meta['config']['bootstrap_on_truncation'])
            expected = identity(manifest, pipeline=pipeline, stage='fqi_fit', method=method,
                dataset=dataset, variant=f'epoch_{epoch}', smoke=smoke,
                seeds=[580001+epoch*100+i for i in range(len(meta['config']['actions']))],
                effective={'learning': meta['config'], 'extra_trees': asdict(ExtraTreesConfig(**(extra_trees_config or {})))}
                    if method == 'extra_trees' else meta['config'], hashes={'data': meta['npz_sha256'], 'catalog': encoder.catalog_hash,
                    'targets': array_hash(target), 'previous_model': previous_hash})
            previous, fit_record = fit_iteration(output/method/f'iteration_{epoch}', expected, method,
                transitions, encoder, meta['config'], previous, epoch, smoke=smoke, extra_trees_config=extra_trees_config)
            previous_hash = fit_record['model_sha256']
    for method in METHODS:
        for epoch in epochs:
            folder = output/method/f'iteration_{epoch}'
            from experiments.pong_analysis_runtime import checked_read
            with unit_lock(folder):
                fit_id = checked_read(folder/'identity.json')
                unit = WorkUnit(folder, fit_id)
                saved = unit.load()
                model = unit.pickle(saved, 'model')
            expected = identity(manifest, pipeline=pipeline, stage='fqi_evaluation', method=method,
                dataset=dataset, variant=f'epoch_{epoch}', smoke=smoke, seeds=seeds,
                effective={'environment': meta.get('common_protocol', protocol['environments'][pipeline])},
                hashes={'data': meta['npz_sha256'], 'targets': fit_id['hashes']['targets'], 'catalog': encoder.catalog_hash},
                evaluation={'model': saved['result']['model_sha256'], 'horizon': horizon, 'policy': 'greedy_random_ties'})
            rows = evaluate_model(model, factory, seeds, horizon, cfg['max_seconds_per_episode'],
                output=output/method/f'evaluation_{epoch}', expected=expected,
                extend_seconds=extend_seconds, extension_id=extension_id)
            atomic_json(output/method/f'eval_{epoch}.json', {'rows': rows, 'identity': expected})
            print(pipeline, method, epoch, 'completed' if all(r['status']=='completed' for r in rows) else 'partial', flush=True)
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
    atomic_json(path, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', type=Path, required=True)
    parser.add_argument('--pipeline', choices=('ocatari','visual'), required=True)
    parser.add_argument('--stage', choices=('regression','fqi'), required=True)
    parser.add_argument('--dataset', choices=('pilot', 'common', 'assisted'), default='pilot')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--dataset-folder', type=Path, help='Explicit independent portable dataset directory')
    parser.add_argument("--extra-trees-config", type=Path, help="Explicit nominal ExtraTrees settings; recorded in unit identity")
    extension_arguments(parser)
    args = parser.parse_args()
    extra_options = read(args.extra_trees_config) if args.extra_trees_config else {}
    ExtraTreesConfig(**extra_options)
    manifest = load_manifest(args.analysis, writable=True)
    if args.dataset in ('common', 'assisted') and args.dataset_folder is None:
        parser.error('New common/assisted data require an explicit --dataset-folder')
    output = mode_root(args.analysis, args.pipeline, args.stage+'_'+args.dataset, args.smoke)
    if args.stage == 'regression':
        with unit_lock(output):
            report = regression_benchmark(args.analysis, args.pipeline, smoke=args.smoke, output=output, manifest=manifest, dataset_folder=args.dataset_folder, extra_trees_config=extra_options)
            atomic_json(output / 'report.json', report)
    else:
        run_fqi(args.analysis, args.pipeline, output, smoke=args.smoke, dataset=args.dataset,
                extend_seconds=args.extend_seconds, extension_id=args.extension_id, dataset_folder=args.dataset_folder, extra_trees_config=extra_options)


if __name__ == '__main__':
    main()
