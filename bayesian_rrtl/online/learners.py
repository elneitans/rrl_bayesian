"""B8/B9: common block learner, explicit estimator contracts and capabilities."""
from collections import OrderedDict
import copy
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from types import SimpleNamespace
import time

import numpy as np

from bayesian_rrtl.agent import (ActionDataset, BayesianQAgent, BayesianQModel, FitRecord,
                                calibration_design)
from bayesian_rrtl.baseline_adapter import BaselineLearnerConfig, RelationalBaselineAgent
from bayesian_rrtl.controls import PointTreeModel, fit_cart
from bayesian_rrtl.ensemble import BARTRegressor
from bayesian_rrtl.priors import FixedScale, calibrate_noise
from bayesian_rrtl.q_config import BayesianQConfig
from bayesian_rrtl.replay import concatenate_states
from bayesian_rrtl.targets import frozen_q_targets
from experiments.pong_analysis_runtime import bytes_hash, digest


def sklearn_version():
    try:
        return version('scikit-learn')
    except PackageNotFoundError:
        return None


@dataclass(frozen=True)
class LearningConfig:
    actions: tuple[str, ...]
    seed: int
    gamma: float = .99
    reward_min: float = -1.
    reward_max: float = 1.
    replay_capacity: int = 50000
    batch_size_per_action: int = 128
    fit_interval: int = 512
    warmup_interactions: int = 512
    epsilon_init: float = 1.
    epsilon_min: float = .1
    epsilon_decay_steps: int = 50000
    bootstrap_on_truncation: bool = True
    validation_seeds: tuple[int, ...] = ()
    test_seeds: tuple[int, ...] = ()
    resample_repeated_actions: bool = False
    action_buffer_capacity: int = 10

    def __post_init__(self):
        for key in ('actions', 'validation_seeds', 'test_seeds'):
            object.__setattr__(self, key, tuple(getattr(self, key)))
        if not self.actions or len(set(self.actions)) != len(self.actions):
            raise ValueError('Invalid actions')
        for key in ('replay_capacity', 'batch_size_per_action', 'fit_interval',
                    'epsilon_decay_steps', 'action_buffer_capacity'):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f'Invalid {key}')
        if type(self.warmup_interactions) is not int or self.warmup_interactions < 0:
            raise ValueError('Invalid warmup')
        if not 0 < self.epsilon_min <= self.epsilon_init <= 1 or not 0 <= self.gamma < 1:
            raise ValueError('Invalid policy/discount')
        if self.resample_repeated_actions or type(self.bootstrap_on_truncation) is not bool:
            raise ValueError('Common policy disables action resampling; bootstrap must be boolean')
        seeds = (self.seed, *self.validation_seeds, *self.test_seeds)
        if len(set(seeds)) != len(seeds) or any(type(s) is not int or not 0 <= s < 2**32 for s in seeds):
            raise ValueError('Invalid or overlapping seeds')
        self.scale

    @property
    def scale(self):
        return FixedScale.from_rewards(self.reward_min, self.reward_max, self.gamma)

    @property
    def config_hash(self):
        return digest(asdict(self))


@dataclass(frozen=True)
class CARTConfig:
    max_depth: int = 4
    min_child: int = 2
    criterion: str = 'SSE'

    def __post_init__(self):
        if (type(self.max_depth) is not int or self.max_depth < 0 or
                type(self.min_child) is not int or self.min_child < 1 or self.criterion != 'SSE'):
            raise ValueError('Invalid CART configuration')


@dataclass(frozen=True)
class ExtraTreesConfig:
    n_estimators: int = 20
    max_depth: int = 4
    min_samples_leaf: int = 2
    n_jobs: int = 1
    sklearn_version: str = '1.9.1'
    encoding: str = 'nominal_missing_v1'

    def __post_init__(self):
        if any(type(v) is not int or v < 1 for v in
               (self.n_estimators, self.max_depth, self.min_samples_leaf)) or self.n_jobs != 1:
            raise ValueError('Invalid ExtraTrees configuration; n_jobs must be 1')
        if self.encoding != 'nominal_missing_v1' or not self.sklearn_version:
            raise ValueError('Invalid nominal encoding/dependency version')


def nominal_indicators(batch):
    """Frozen columns: catalog order, missing first, then every observed category."""
    return np.column_stack([(batch.values[:, j] == level).astype(np.float32)
        for j, key in enumerate(batch.catalog)
        for level in (-1, *(range(3) if key.type == 'comparative' else range(2)))])


@dataclass(frozen=True)
class NominalExtraTrees:
    estimator: object
    catalog_hash: str
    config: ExtraTreesConfig

    def predict_mean(self, batch):
        if batch.catalog_hash != self.catalog_hash or sklearn_version() != self.config.sklearn_version:
            raise ValueError('ExtraTrees catalog/dependency mismatch')
        return self.estimator.predict(nominal_indicators(batch))

    def diagnostics(self):
        return {'estimator': 'ExtraTrees', 'config': asdict(self.config),
                'trees': len(self.estimator.estimators_),
                'leaves': int(sum(t.get_n_leaves() for t in self.estimator.estimators_)),
                'max_depth': int(max(t.get_depth() for t in self.estimator.estimators_))}


class PredictionCache:
    """LRU measured byte bound; stores only reference predictions, never consumes RNG."""
    def __init__(self, max_bytes=1048576):
        if type(max_bytes) is not int or max_bytes < 0:
            raise ValueError('Invalid cache memory limit')
        self.max_bytes = max_bytes
        self.clear()

    def clear(self):
        self.entries = OrderedDict()
        self.bytes = self.hits = self.misses = 0
        self.current_model = None

    def predict(self, model, batch):
        if self.current_model is not model:
            self.clear()
            self.current_model = model
        # Cache whole reference batches: no change to posterior reduction arithmetic.
        key = (model.model_id, batch.catalog_hash, batch.values.shape,
               batch.values.tobytes(), batch.observed.tobytes())
        if key in self.entries:
            self.hits += 1
            value, size = self.entries.pop(key)
            self.entries[key] = value, size
            return value.copy()
        self.misses += 1
        value = model.predict(batch)
        # Includes Python container overhead conservatively, plus all key/value bytes.
        size = len(key[1]) + len(key[3]) + len(key[4]) + value.nbytes + 1024
        if size <= self.max_bytes:
            while self.bytes + size > self.max_bytes:
                _, (_, old_size) = self.entries.popitem(last=False)
                self.bytes -= old_size
            self.entries[key] = value.copy(), size
            self.bytes += size
        return value


class OnlineBatchAgent(BayesianQAgent):
    """All batch methods share observe, replay selection, targets and atomic publication."""
    def __init__(self, config, encoder, *, learner, estimator, cache_bytes=1048576):
        if learner not in ('cart', 'bart', 'extra_trees'):
            raise ValueError('Unknown batch learner')
        self.learner_id = learner
        self.estimator_config = estimator
        if learner == 'cart':
            CARTConfig(**estimator)
        elif learner == 'extra_trees':
            c = ExtraTreesConfig(**estimator)
            if sklearn_version() != c.sklearn_version:
                raise ValueError('Incompatible ExtraTrees dependency')
        else:
            # Explicit composition, not a reinterpretation of historic flat config.
            BayesianQConfig(**asdict(config), **estimator)
        super().__init__(config, encoder)
        self.cache = PredictionCache(cache_bytes)
        self.phase_seconds = {'selection_targets': 0., 'fit': 0., 'prediction': 0., 'encoding': 0.}
        self.fit_count = 0
        self.last_fit_seconds = 0.

    def encode_state(self, state):
        start = time.perf_counter()
        batch = super().encode_state(state)
        self.phase_seconds['encoding'] += time.perf_counter()-start
        return batch

    def predict_q(self, batch):
        start = time.perf_counter()
        result = self.cache.predict(self.model, batch)
        self.phase_seconds['prediction'] += time.perf_counter()-start
        return result

    def select_action(self, state, *, explore=True):
        batch = self.encode_state(state)
        rng = self.exploration_rng
        if explore and rng.random() < self.epsilon:
            return int(rng.integers(len(self.config.actions)))
        q = self.predict_q(batch)[0]
        return int(rng.choice(np.flatnonzero(q == q.max())))

    def fit_action(self, dataset, rng, calibration):
        c, e = self.config, self.estimator_config
        if self.learner_id == 'cart':
            tree = fit_cart(dataset.batch, dataset.targets, CARTConfig(**e))
            return PointTreeModel(tree, self.encoder.catalog_hash, c.scale), None
        if self.learner_id == 'extra_trees':
            from sklearn.ensemble import ExtraTreesRegressor
            options = ExtraTreesConfig(**e)
            model = ExtraTreesRegressor(n_estimators=options.n_estimators, max_depth=options.max_depth,
                min_samples_leaf=options.min_samples_leaf, n_jobs=1,
                random_state=int(rng.integers(2**32))).fit(nominal_indicators(dataset.batch), dataset.targets)
            return NominalExtraTrees(model, self.encoder.catalog_hash, options), None
        bart = BayesianQConfig(**asdict(c), **e)
        if calibration is None:
            calibration = calibrate_noise(c.scale.normalize(dataset.targets), calibration_design(dataset.batch),
                nu=bart.nu, quantile=bart.noise_quantile, variance_floor=bart.variance_floor)
        posterior = BARTRegressor(kernel=bart.kernel, m=bart.m, tree_prior=bart.tree_prior,
            scale=c.scale, k=bart.k, noise_prior=calibration.prior).fit(dataset.batch, dataset.targets,
                rng, burn_in=bart.burn_in, draws=bart.draws)
        return posterior, calibration

    def fit_if_due(self, *, force=False):
        c = self.config
        if (self.interactions < c.warmup_interactions or not len(self.replay)
                or self.interactions == self.last_fit_step
                or (not force and self.interactions-self.last_fit_step < c.fit_interval)):
            return None
        target = self.model
        replay_rng, rngs = copy.deepcopy(self.replay_rng), copy.deepcopy(self.mcmc_rngs)
        started = time.perf_counter()
        datasets = []
        for action in range(len(c.actions)):
            transitions = self.replay.sample(action, c.batch_size_per_action, replay_rng)
            batch = concatenate_states([t.state for t in transitions], self.encoder)
            y = frozen_q_targets(transitions, target, self.encoder, gamma=c.gamma,
                                 bootstrap_on_truncation=c.bootstrap_on_truncation)
            ids = tuple(t.transition_id for t in transitions)
            sha = bytes_hash(batch.catalog_hash.encode() + np.asarray(ids, dtype='<i8').tobytes()
                             + batch.values.tobytes() + batch.observed.tobytes() + y.astype('<f8').tobytes())
            datasets.append(ActionDataset(action, ids, batch, y, sha))
        selection_seconds = time.perf_counter()-started
        started = time.perf_counter()
        models, calibrations = list(target.posteriors), list(self.calibrations)
        updated, skipped, details = [], [], {}
        for data in datasets:
            action = data.action
            if not len(data.batch):
                skipped.append(action)
                details[c.actions[action]] = {'updated': False, 'reason': 'no_replay_data'}
                continue
            models[action], calibrations[action] = self.fit_action(data, rngs[action], calibrations[action])
            details[c.actions[action]] = {**models[action].diagnostics(), 'updated': True,
                                         'samples': len(data.batch), 'data_hash': data.data_hash}
            updated.append(action)
        fit_seconds = time.perf_counter()-started
        model = BayesianQModel(c.actions, tuple(models), self.encoder.catalog_hash, target.model_id+1)
        record = FitRecord(model.model_id, target.model_id, self.interactions,
            digest({'learning': asdict(c), 'estimator': self.estimator_config, 'learner': self.learner_id}),
            tuple(datasets), tuple(updated), tuple(skipped), details)
        # Commit once; failed fits leave previous model, calibration and RNG intact.
        self.model, self.target_model = model, target
        self.calibrations, self.replay_rng, self.mcmc_rngs = tuple(calibrations), replay_rng, rngs
        self.last_fit_step, self.last_fit = self.interactions, record
        self.fit_count += 1
        self.last_fit_seconds = selection_seconds+fit_seconds
        self.phase_seconds['selection_targets'] += selection_seconds
        self.phase_seconds['fit'] += fit_seconds
        self.fit_history = [{'fit_id': model.model_id, 'interaction': self.interactions, 'actions': details}]
        self.cache.clear()
        return record


def _baseline(config, encoder, estimator, cache_bytes):
    if set(estimator) & {'seed', 'gamma', 'bootstrap_on_truncation'}:
        raise ValueError('Baseline shared fields belong in learning config')
    return RelationalBaselineAgent(config.actions, encoder, BaselineLearnerConfig(**estimator,
        seed=config.seed, gamma=config.gamma, bootstrap_on_truncation=config.bootstrap_on_truncation))


def _batch(kind):
    return lambda config, encoder, estimator, cache_bytes: OnlineBatchAgent(
        config, encoder, learner=kind, estimator=estimator, cache_bytes=cache_bytes)


LEARNERS = {
    'corrected_common': SimpleNamespace(batch=False, posterior=False, constructor=_baseline),
    'cart': SimpleNamespace(batch=True, posterior=False, constructor=_batch('cart')),
    'bart': SimpleNamespace(batch=True, posterior=True, constructor=_batch('bart')),
    'extra_trees': SimpleNamespace(batch=True, posterior=False, constructor=_batch('extra_trees')),
}


def make_agent(learner, config, encoder, estimator, *, cache_bytes=1048576):
    if learner not in LEARNERS:
        raise ValueError(f'Unknown learner: {learner}')
    return LEARNERS[learner].constructor(config, encoder, estimator, cache_bytes)


def learner_metrics(agent, learner, batch):
    capability = LEARNERS[learner]
    if not capability.batch:
        def nodes(root, depth=0):
            yield root, depth
            for child in root.children:
                yield from nodes(child, depth+1)
        result = {'kind': 'rrtl_corrected_common_v1', 'splits': agent.n_splits,
                  'fallback': copy.deepcopy(agent.query_counts), 'complexity': {}}
        for name, root in agent.root_nodes.items():
            entries = list(nodes(root))
            result['complexity'][name] = {'leaves': sum(not n.children for n, _ in entries),
                                         'max_depth': max(d for _, d in entries)}
        return result
    result = {'kind': learner, 'fits': agent.fit_count, 'replay_size': len(agent.replay),
              'phase_seconds': dict(agent.phase_seconds), 'cache_bytes': agent.cache.bytes, 'complexity': {}}
    for name, model in zip(agent.config.actions, agent.model.posteriors):
        if model is None:
            continue
        if learner == 'cart':
            entries = list(model.tree.walk(batch))
            details = {'leaves': sum(n.is_leaf for _, n, _ in entries),
                       'max_depth': max(len(p) for p, _, _ in entries)}
        elif capability.posterior:
            details = {'trees': model.m, 'draws': len(model.draws),
                       'leaves_last_draw': model.trace[-1].total_leaves,
                       'max_depth_last_draw': model.trace[-1].max_depth,
                       'diagnostics': model.diagnostics()}
        else:
            details = model.diagnostics()
        result['complexity'][name] = details
    return result
