"""Block fitted Q with a frozen target and independently restarted BART chains."""

from collections import deque
import copy
from dataclasses import dataclass
import hashlib

import numpy as np

from .encoding import EncodedBatch
from .ensemble import BARTPosterior, BARTRegressor
from .priors import calibrate_noise
from .replay import ReplayBuffer, concatenate_states
from .targets import bootstrap_mask, frozen_q_targets


@dataclass(frozen=True)
class BayesianQModel:
    actions: tuple[str, ...]
    posteriors: tuple[BARTPosterior | None, ...]
    catalog_hash: str
    model_id: int = 0

    def predict(self, batch):
        if batch.catalog_hash != self.catalog_hash:
            raise ValueError('Prediction catalog mismatch')
        # Explicit Q=0 initialization; None means no posterior, NOT zero uncertainty.
        return np.column_stack([np.zeros(len(batch)) if p is None else p.predict_mean(batch)
                                for p in self.posteriors])

    def predict_latent_draws(self, batch, action):
        posterior = self.posteriors[action]
        if posterior is None:
            raise ValueError('Action has no fitted posterior')
        return posterior.predict_latent_draws(batch)


@dataclass(frozen=True)
class ActionDataset:
    action: int
    transition_ids: tuple[int, ...]
    batch: EncodedBatch
    targets: np.ndarray
    data_hash: str


@dataclass(frozen=True)
class FitRecord:
    fit_id: int
    target_model_id: int
    interaction: int
    config_hash: str
    datasets: tuple[ActionDataset, ...]
    updated_actions: tuple[int, ...]
    skipped_actions: tuple[int, ...]
    diagnostics: dict


def calibration_design(batch):
    """Nominal one-hot design with intercept, including missingness (no ordinal codes)."""
    columns = [np.ones(len(batch))]
    for j in range(len(batch.catalog)):
        # Drop one observed level to avoid the intercept alias.
        levels = np.unique(batch.values[:, j])
        columns.extend((batch.values[:, j] == level).astype(float) for level in levels[1:])
    return np.column_stack(columns)


class BayesianQAgent:
    def __init__(self, config, encoder):
        self.config = config
        self.encoder = encoder
        self.replay = ReplayBuffer(config.replay_capacity)
        streams = np.random.SeedSequence(config.seed).spawn(3 + len(config.actions))
        self.environment_rng, self.exploration_rng, self.replay_rng = [
            np.random.default_rng(s) for s in streams[:3]]
        self.mcmc_rngs = tuple(np.random.default_rng(s) for s in streams[3:])
        self.model = BayesianQModel(config.actions, (None,) * len(config.actions), encoder.catalog_hash)
        self.target_model = self.model
        self.calibrations = (None,) * len(config.actions)
        self.interactions = self.last_fit_step = 0
        self.last_fit = None
        self.fit_history = []
        self.training_seeds = []
        self.action_buffer = deque(maxlen=config.action_buffer_capacity)

    @property
    def epsilon(self):
        c = self.config
        return max(c.epsilon_min, c.epsilon_init * (c.epsilon_min / c.epsilon_init)**(
            self.interactions / c.epsilon_decay_steps))

    def episode_seed(self):
        reserved = set(self.config.validation_seeds + self.config.test_seeds)
        while True:
            seed = int(self.environment_rng.integers(0, 2**32))
            if seed not in reserved:
                self.training_seeds.append(seed)
                return seed

    def encode_state(self, state):
        batch = state if isinstance(state, EncodedBatch) else self.encoder.transform([state])
        if len(batch) != 1 or batch.catalog_hash != self.encoder.catalog_hash:
            raise ValueError('State must be one row of the configured catalog')
        return batch

    def select_action(self, state, *, explore=True):
        batch = self.encode_state(state)
        rng = self.exploration_rng
        if explore and rng.random() < self.epsilon:
            action = int(rng.integers(len(self.config.actions)))
        else:
            q = self.model.predict(batch)[0]
            action = int(rng.choice(np.flatnonzero(q == q.max())))
        if self.config.resample_repeated_actions:
            # Same convention as the historical buffer: append the proposal before resampling.
            self.action_buffer.append(action)
            if len(self.action_buffer) == self.action_buffer.maxlen and len(set(self.action_buffer)) == 1:
                action = int(rng.integers(len(self.config.actions)))
        return action

    def observe(self, state, action, reward, next_state, terminated, truncated, *, raw_reward,
                episode_id, step_id):
        if type(action) is not int or not 0 <= action < len(self.config.actions):
            raise ValueError('Unknown action')
        mask = bootstrap_mask(terminated, truncated,
                              bootstrap_on_truncation=self.config.bootstrap_on_truncation,
                              final_observation_available=next_state is not None)
        if mask and next_state is None:
            raise ValueError('Bootstrap requires the final next observation')
        # Terminal next observations can be absent, invalid or outside the catalog.
        next_batch = self.encode_state(next_state) if mask else None
        transition = self.replay.append(state=self.encode_state(state), action=action,
                                        reward=float(reward), raw_reward=float(raw_reward),
                                        next_state=next_batch, terminated=terminated, truncated=truncated,
                                        episode_id=episode_id, step_id=step_id)
        self.interactions += 1
        return transition

    def make_regressor(self, action, calibration):
        c = self.config
        return BARTRegressor(kernel=c.kernel, m=c.m, tree_prior=c.tree_prior, scale=c.scale,
                             k=c.k, noise_prior=calibration.prior)

    def fit_if_due(self, *, force=False):
        if self.interactions < self.config.warmup_interactions:
            return None
        if not len(self.replay) or self.interactions == self.last_fit_step:
            return None
        if not force and self.interactions - self.last_fit_step < self.config.fit_interval:
            return None
        c = self.config
        target = self.model
        # Stage RNG and models too: a failed action fit must not partially publish a block.
        replay_rng = copy.deepcopy(self.replay_rng)
        mcmc_rngs = copy.deepcopy(self.mcmc_rngs)
        datasets = []
        for action in range(len(c.actions)):
            transitions = self.replay.sample(action, c.batch_size_per_action, replay_rng)
            batch = concatenate_states([t.state for t in transitions], self.encoder)
            y = frozen_q_targets(transitions, target, self.encoder, gamma=c.gamma,
                                 bootstrap_on_truncation=c.bootstrap_on_truncation)
            ids = tuple(t.transition_id for t in transitions)
            digest = hashlib.sha256(batch.catalog_hash.encode() + np.asarray(ids, dtype='<i8').tobytes()
                                    + batch.values.tobytes() + batch.observed.tobytes()
                                    + y.astype('<f8').tobytes()).hexdigest()
            datasets.append(ActionDataset(action, ids, batch, y, digest))
        posteriors = list(target.posteriors)
        calibrations = list(self.calibrations)
        updated, skipped, diagnostics = [], [], {}
        for dataset in datasets:
            action = dataset.action
            if not len(dataset.batch):
                skipped.append(action)
                diagnostics[c.actions[action]] = {'updated': False, 'reason': 'no_replay_data'}
                continue
            if calibrations[action] is None:
                calibrations[action] = calibrate_noise(c.scale.normalize(dataset.targets),
                    calibration_design(dataset.batch), nu=c.nu, quantile=c.noise_quantile,
                    variance_floor=c.variance_floor)
            posterior = self.make_regressor(action, calibrations[action]).fit(
                dataset.batch, dataset.targets, mcmc_rngs[action], burn_in=c.burn_in, draws=c.draws)
            posteriors[action] = posterior
            updated.append(action)
            diagnostics[c.actions[action]] = dict(posterior.diagnostics(), updated=True,
                samples=len(dataset.batch), data_hash=dataset.data_hash,
                max_abs_q=float(np.abs(posterior.predict_mean(dataset.batch)).max()),
                replay_coverage=sum(t.action == action for t in self.replay.transitions),
                noise_calibration=calibrations[action].method)
        model = BayesianQModel(c.actions, tuple(posteriors), self.encoder.catalog_hash, target.model_id + 1)
        record = FitRecord(model.model_id, target.model_id, self.interactions, c.config_hash,
                           tuple(datasets), tuple(updated), tuple(skipped), diagnostics)
        self.model, self.target_model = model, target
        self.calibrations = tuple(calibrations)
        self.replay_rng, self.mcmc_rngs = replay_rng, mcmc_rngs
        self.last_fit_step, self.last_fit = self.interactions, record
        self.fit_history.append({'fit_id': record.fit_id, 'target_model_id': record.target_model_id,
                                 'interaction': record.interaction, 'config_hash': record.config_hash,
                                 'actions': diagnostics})
        return record

    def evaluate(self, environment_factory, seeds):
        seeds = tuple(seeds)
        if set(seeds) & set(self.training_seeds):
            raise ValueError('Evaluation seeds overlap training')
        private = copy.deepcopy(self)
        environment = environment_factory()
        rows = []
        try:
            for seed in seeds:
                private.exploration_rng = np.random.default_rng(seed)
                private.action_buffer.clear()
                state = environment.reset(seed=seed)
                total, steps = 0.0, 0
                while True:
                    action = private.select_action(state, explore=False)
                    state, reward, raw_reward, terminated, truncated = environment.step(action)
                    total += raw_reward
                    steps += 1
                    if terminated or truncated:
                        break
                rows.append({'seed': seed, 'return': total, 'steps': steps,
                             'terminated': terminated, 'truncated': truncated})
        finally:
            environment.close()
        return rows
