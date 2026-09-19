"""Pong comparison factories, shared reporting and frozen evaluation (plan B).

The H5 collector and learning algorithms are unchanged. Protocol freezing,
campaign execution and checkpoint formats belong to subsequent plan stages.
"""

import copy
from dataclasses import asdict, dataclass, replace

import numpy as np

from .agent import BayesianQAgent
from .atari import AtariConfig, AtariRelationalEnvironment
from .baseline_adapter import BaselineLearnerConfig, RelationalBaselineAgent
from .game_registry import get_game_spec
from .q_config import BayesianQConfig
from .training import QTrainingSession


@dataclass(frozen=True)
class ComparisonPolicyConfig:
    epsilon_init: float = 1.
    epsilon_min: float = .1
    epsilon_decay_steps: int = 50_000
    validation_seeds: tuple[int, ...] = (100001, 100002)
    test_seeds: tuple[int, ...] = (200001, 200002)

    def __post_init__(self):
        for name in ('validation_seeds', 'test_seeds'):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if (self.epsilon_init != 1. or self.epsilon_min != .1
                or type(self.epsilon_decay_steps) is not int or self.epsilon_decay_steps != 50_000):
            raise ValueError('Comparison requires epsilon 1 -> .1 over 50000 decisions')
        self.validate_seed(None)

    def validate_seed(self, seed):
        seeds = self.validation_seeds + self.test_seeds + (() if seed is None else (seed,))
        if (any(type(s) is not int or not 0 <= s < 2**32 for s in seeds)
                or len(set(seeds)) != len(seeds)):
            raise ValueError('Training, validation and test seeds must be disjoint uint32 values')

    def epsilon(self, interactions):
        # Same closed form as BayesianQAgent.epsilon; no multiplicative drift.
        return max(self.epsilon_min, self.epsilon_init * (self.epsilon_min / self.epsilon_init)**(
            interactions / self.epsilon_decay_steps))

    def validate_bart(self, config):
        self.validate_seed(config.seed)
        if any(getattr(config, key) != value for key, value in asdict(self).items()):
            raise ValueError('BART policy/seed configuration differs from comparison policy')
        if config.resample_repeated_actions:
            raise ValueError('Comparison disables repeated-action resampling')


@dataclass(frozen=True)
class PongEnvironmentFactory:
    config: AtariConfig

    def __post_init__(self):
        # Resolve additive defaults once so every created instance uses the full contract.
        config = AtariConfig(**self.config.resolved_dict())
        backend = config.extractor
        if (config.game != 'Pong' or backend not in ('legacy_visual_pong_v1', 'ocatari_ram_v1')
                or config.representation != 'comparative' or config.frameskip != 4
                or config.repeat_action_probability != 0
                or config.include_incomplete_states != (backend == 'ocatari_ram_v1')):
            raise ValueError('Environment differs from the declared Pong comparison conditions')
        object.__setattr__(self, 'config', config)

    def __call__(self):
        return AtariRelationalEnvironment(self.config)

    def with_horizon(self, horizon):
        return type(self)(replace(self.config, max_episode_steps=horizon))


def _validate_environment(agent, environment, *, training_environment=None):
    actions = agent.actions if isinstance(agent, RelationalBaselineAgent) else agent.config.actions
    if tuple(environment.actions) != tuple(actions) or environment.encoder().to_dict() != agent.encoder.to_dict():
        raise ValueError('Environment actions/encoder differ from learner')
    if training_environment is not None and hasattr(training_environment, 'config'):
        expected = training_environment.config.resolved_dict()
        actual = environment.config.resolved_dict()
        # Evaluation may have a different horizon, but not a different pipeline.
        expected.pop('max_episode_steps')
        actual.pop('max_episode_steps')
        if actual != expected:
            raise ValueError('Evaluation environment differs from training semantics')


def _mean(values):
    return float(np.mean(values)) if values else None


def _return_scope(terminated, truncated):
    return 'complete_game' if terminated else 'horizon_limited' if truncated else 'partial'


def evaluation_summary(rows, horizon):
    """Episode-level return dispersion (population SD and linear-quartile IQR)."""
    returns = [row['return'] for row in rows]
    return {'schema': 'pong_evaluation_v2', 'episodes': rows, 'horizon': horizon,
            'mean_raw_return': _mean(returns), 'median_raw_return': float(np.median(returns)),
            'std_raw_return': float(np.std(returns, ddof=0)),
            'iqr_raw_return': float(np.percentile(returns, 75) - np.percentile(returns, 25)),
            'truncation_fraction': sum(r['truncated'] for r in rows) / len(rows),
            'completed_games': sum(r['terminated'] for r in rows),
            'mean_completed_game_raw_return': _mean([r['return'] for r in rows if r['terminated']])}


class ComparisonSessionReporting:
    """Common transition/episode schema; only ended episodes enter episode_history."""

    def _init_reporting(self, policy_config, environment_factory):
        if self.agent.interactions:
            raise ValueError('A fresh session requires an untrained agent; restore needs its collector')
        _validate_environment(self.agent, self.environment)
        self.policy_config = policy_config
        self.environment_factory = environment_factory
        self.episode_history = []

    @property
    def interactions(self):
        return self.agent.interactions

    def _record_common(self, row, episode_seed, learner_metrics):
        row.update(episode_seed=episode_seed, episode_end=bool(self.done),
                   return_scope=_return_scope(row['terminated'], row['truncated']),
                   learner_metrics=learner_metrics)
        if self.done:
            self.episode_history.append(self._episode_row(row))
        return row

    @staticmethod
    def _episode_row(row):
        return {key: row[key] for key in ('episode_id', 'episode_seed', 'interaction',
                'transition_id', 'episode_return', 'raw_episode_return', 'terminated',
                'truncated', 'episode_end', 'return_scope')} | {'steps': row['step_id'] + 1}

    def episode_rows(self):
        rows = copy.deepcopy(self.episode_history)
        if self.history and not self.done:
            rows.append(self._episode_row(self.history[-1]))
        return rows

    def metrics(self):
        completed = [r['raw_episode_return'] for r in self.episode_history if r['terminated']]
        limited = [r['raw_episode_return'] for r in self.episode_history
                   if r['truncated'] and not r['terminated']]
        return {'interactions': self.interactions, 'ended_episodes': len(self.episode_history),
                'completed_games': len(completed), 'horizon_limited_episodes': len(limited),
                'mean_completed_game_raw_return': _mean(completed),
                'mean_horizon_limited_raw_return': _mean(limited),
                'partial_episode': self._episode_row(self.history[-1]) if self.history and not self.done else None,
                'learner': self._learner_metrics()}

    def evaluate(self, seeds, *, horizon=None, environment_factory=None):
        """Greedy frozen copy, private RNG, fresh environment per evaluation seed."""
        seeds = tuple(seeds)
        if (not seeds or len(set(seeds)) != len(seeds)
                or any(type(s) is not int or not 0 <= s < 2**32 for s in seeds)):
            raise ValueError('Evaluation needs unique uint32 seeds')
        training_seeds = (self.training_seeds if isinstance(self.agent, RelationalBaselineAgent)
                          else self.agent.training_seeds)
        if set(seeds) & set(training_seeds):
            raise ValueError('Evaluation seeds overlap training')
        factory = environment_factory or self.environment_factory
        if factory is None:
            raise ValueError('Evaluation requires a fresh environment factory')
        if horizon is None and isinstance(factory, PongEnvironmentFactory):
            horizon = factory.config.max_episode_steps
        if type(horizon) is not int or horizon < 1:
            raise ValueError('Evaluation horizon must be a positive integer')
        if isinstance(factory, PongEnvironmentFactory):
            factory = factory.with_horizon(horizon)
        private = copy.deepcopy(self.agent)
        rows = []
        for episode_id, seed in enumerate(seeds):
            environment = factory()
            if environment is self.environment:
                raise ValueError('Evaluation must not reuse the training environment')
            try:
                _validate_environment(private, environment, training_environment=self.environment)
                rng = np.random.default_rng(seed)
                if isinstance(private, BayesianQAgent):
                    private.exploration_rng = rng
                    private.action_buffer.clear()
                state = environment.reset(seed=seed)
                total = 0.
                points_won = points_lost = 0.
                for step in range(1, horizon + 1):
                    if isinstance(private, RelationalBaselineAgent):
                        action = private.select_action(state, epsilon=0., rng=rng)
                    else:
                        action = private.select_action(state, explore=False)
                    state, _, raw, terminated, truncated = environment.step(action)
                    total += raw
                    points_won += max(float(raw), 0.)
                    points_lost += max(-float(raw), 0.)
                    if terminated or truncated:
                        break
                truncated = bool(truncated or (step == horizon and not terminated))
                rows.append({'episode_id': episode_id, 'seed': seed, 'return': total, 'steps': step,
                             'terminated': terminated, 'truncated': truncated,
                             'points_won': points_won, 'points_lost': points_lost,
                             'return_scope': _return_scope(terminated, truncated), 'epsilon': 0.})
            finally:
                environment.close()
        return evaluation_summary(rows, horizon)


class BayesianComparisonSession(ComparisonSessionReporting, QTrainingSession):
    """Decorate H5 logs; super().step() retains collection, replay and fit scheduling."""

    def __init__(self, agent, environment, policy_config=ComparisonPolicyConfig(), *, environment_factory=None):
        policy_config.validate_bart(agent.config)
        super().__init__(agent, environment)
        self._init_reporting(policy_config, environment_factory)
        self._checkpoint_ready = True

    @property
    def epsilon(self):
        return self.agent.epsilon

    def step(self):
        if not self._checkpoint_ready:
            raise ValueError('Incomplete update; restore the last completed checkpoint')
        self._checkpoint_ready = False
        before = self.agent.model.model_id
        row = super().step()
        fitted = self.agent.model.model_id != before
        specific = {'kind': 'bart', 'fit_id': self.agent.model.model_id, 'fit_performed': fitted,
                    'fit_diagnostics': copy.deepcopy(self.agent.last_fit.diagnostics) if fitted else None}
        row = self._record_common(row, self.agent.training_seeds[-1], specific)
        self._checkpoint_ready = True
        return row

    def _learner_metrics(self):
        return {'kind': 'bart', 'fits': self.agent.model.model_id, 'last_fit_step': self.agent.last_fit_step,
                'replay_size': len(self.agent.replay), 'fit_history': copy.deepcopy(self.agent.fit_history)}

    def save(self, path):
        from .comparison_checkpoint import save_session
        return save_session(self, path)

    @classmethod
    def load(cls, *args, **kwargs):
        from .comparison_checkpoint import load_session
        return load_session(*args, **kwargs)


def make_comparison_session(learner, environment_config, learner_config, *, policy_config=None):
    """Validate before opening ALE; caller owns session.environment and closes it."""
    factory = PongEnvironmentFactory(environment_config)
    spec = get_game_spec(factory.config.game, factory.config.extractor)
    if learner == 'bart':
        if not isinstance(learner_config, BayesianQConfig):
            raise ValueError('BART requires BayesianQConfig')
        policy_config = policy_config or ComparisonPolicyConfig(
            validation_seeds=learner_config.validation_seeds, test_seeds=learner_config.test_seeds)
        policy_config.validate_bart(learner_config)
        if (learner_config.actions != spec.actions
                or (learner_config.reward_min, learner_config.reward_max) != factory.config.reward_bounds):
            raise ValueError('BART actions/reward bounds differ from the environment')
    elif learner == 'corrected_common':
        if not isinstance(learner_config, BaselineLearnerConfig):
            raise ValueError('Baseline requires BaselineLearnerConfig')
        policy_config = policy_config or ComparisonPolicyConfig()
        policy_config.validate_seed(learner_config.seed)
    else:
        raise ValueError('Unknown comparison learner')
    if learner_config.gamma != .99:
        raise ValueError('Pong comparison requires gamma=.99')
    environment = factory()
    try:
        if learner == 'bart':
            agent = BayesianQAgent(learner_config, environment.encoder())
            return BayesianComparisonSession(agent, environment, policy_config, environment_factory=factory)
        from .baseline_training import BaselineTrainingSession
        agent = RelationalBaselineAgent.from_environment(environment, learner_config)
        return BaselineTrainingSession(agent, environment, policy_config, environment_factory=factory)
    except Exception:
        environment.close()
        raise
