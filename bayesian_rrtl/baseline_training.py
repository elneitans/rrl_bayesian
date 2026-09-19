"""Incremental relational collector with H5 policy/environment seed semantics."""

import copy
from dataclasses import asdict

import numpy as np

from .comparison import ComparisonPolicyConfig, ComparisonSessionReporting


class BaselineTrainingSession(ComparisonSessionReporting):
    def __init__(self, agent, environment, policy_config=ComparisonPolicyConfig(), *, environment_factory=None):
        policy_config.validate_seed(agent.config.seed)
        self.agent = agent
        self.environment = environment
        self._init_reporting(policy_config, environment_factory)
        # H5 uses child 0 for episodes and child 1 for exploration. Spawning only
        # these two leaves the same streams; there is no replay or MCMC RNG here.
        environment_seed, exploration_seed = np.random.SeedSequence(agent.config.seed).spawn(2)
        self.environment_rng = np.random.default_rng(environment_seed)
        self.exploration_rng = np.random.default_rng(exploration_seed)
        self.training_seeds = []
        self.state = None
        self.done = True
        self.episode_id = -1
        self.step_id = 0
        self.episode_return = self.raw_episode_return = 0.
        self.history = []
        self._checkpoint_ready = True

    @property
    def epsilon(self):
        return self.policy_config.epsilon(self.interactions)

    def episode_seed(self):
        reserved = set(self.policy_config.validation_seeds + self.policy_config.test_seeds)
        while True:
            seed = int(self.environment_rng.integers(0, 2**32))
            if seed not in reserved:
                self.training_seeds.append(seed)
                return seed

    def step(self):
        if not self._checkpoint_ready:
            raise ValueError('Incomplete update; restore the last completed checkpoint')
        self._checkpoint_ready = False
        row = self._step()
        self._checkpoint_ready = True
        return row

    def _step(self):
        if self.done:
            self.state = self.environment.reset(seed=self.episode_seed())
            self.done = False
            self.episode_id += 1
            self.step_id = 0
            self.episode_return = self.raw_episode_return = 0.
        epsilon = self.epsilon
        action = self.agent.select_action(self.state, epsilon=epsilon, rng=self.exploration_rng)
        next_state, reward, raw_reward, terminated, truncated = self.environment.step(action)
        before = self.agent.n_splits
        td = self.agent.observe(self.state, action, reward, next_state, terminated, truncated)
        self.state = next_state
        self.done = terminated or truncated
        self.step_id += 1
        self.episode_return += reward
        self.raw_episode_return += raw_reward
        row = {'transition_id': self.interactions - 1, 'interaction': self.interactions,
               'episode_id': self.episode_id, 'step_id': self.step_id - 1, 'action': action,
               'reward': reward, 'raw_reward': raw_reward, 'terminated': terminated, 'truncated': truncated,
               'episode_return': self.episode_return, 'raw_episode_return': self.raw_episode_return,
               'epsilon': epsilon}
        self.history.append(row)
        return self._record_common(row, self.training_seeds[-1],
            {'kind': self.agent.learner_id, 'td': asdict(td), 'splits': self.agent.n_splits,
             'split_delta': self.agent.n_splits - before})

    def run(self, interactions):
        if type(interactions) is not int or interactions < 0:
            raise ValueError('Interaction budget must be a nonnegative integer')
        for _ in range(interactions):
            self.step()
        return self.history

    def _learner_metrics(self):
        return {'kind': self.agent.learner_id, 'splits': self.agent.n_splits,
                'query_counts': copy.deepcopy(self.agent.query_counts)}

    def save(self, path):
        from .baseline_checkpoint import save_baseline_checkpoint
        return save_baseline_checkpoint(self, path)

    @classmethod
    def load(cls, path, environment=None, **kwargs):
        from .baseline_checkpoint import load_baseline_checkpoint
        return load_baseline_checkpoint(path, environment, **kwargs)
