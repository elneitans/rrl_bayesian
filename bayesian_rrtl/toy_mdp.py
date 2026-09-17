"""Two-stage episodic MDP with exact Q and full mid-episode snapshots."""

import copy
from dataclasses import dataclass

import numpy as np

from .encoding import RelationKey, RelationalEncoder


@dataclass(frozen=True)
class ToyFact:
    type: str = 'logical'
    value: str = 'False'
    name: str = 'late_stage'
    obj1: str = 'agent_t'
    obj2: str | None = None
    obj3: str | None = None


class TwoStepMDP:
    actions = ('advance', 'exit')

    def __init__(self, *, reward_noise=0.0, max_steps=2):
        if reward_noise < 0 or max_steps < 1:
            raise ValueError('Invalid MDP parameters')
        self.reward_noise = reward_noise
        self.max_steps = max_steps
        self.rng = np.random.default_rng(0)
        self.stage = self.steps = 0
        self.done = True

    @staticmethod
    def encoder():
        return RelationalEncoder([RelationKey('logical', 'late_stage', 'agent_t')])

    @staticmethod
    def state(stage):
        return (ToyFact(value='True' if stage else 'False'),)

    def reset(self, *, seed):
        self.rng = np.random.default_rng(seed)
        self.stage = self.steps = 0
        self.done = False
        return self.state(self.stage)

    def step(self, action):
        if self.done or action not in (0, 1):
            raise ValueError('Invalid action or episode already complete')
        if self.stage == 0 and action == 0:
            reward, terminated = 0.0, False
            self.stage = 1
        else:
            reward = 0.4 if self.stage == 0 else (1.0 if action == 0 else -0.2)
            terminated = True
        reward += float(self.rng.normal(0, self.reward_noise))
        self.steps += 1
        truncated = self.steps >= self.max_steps and not terminated
        self.done = terminated or truncated
        return self.state(self.stage), reward, reward, terminated, truncated

    @staticmethod
    def optimal_q(gamma):
        return np.array([[gamma, 0.4], [1.0, -0.2]])

    def snapshot(self):
        return {'kind': 'two_step_v1', 'reward_noise': self.reward_noise, 'max_steps': self.max_steps,
                'stage': self.stage, 'steps': self.steps, 'done': self.done,
                'rng': copy.deepcopy(self.rng.bit_generator.state)}

    def restore(self, state):
        if state['kind'] != 'two_step_v1' or (state['reward_noise'], state['max_steps']) != (
                self.reward_noise, self.max_steps):
            raise ValueError('Environment snapshot mismatch')
        self.stage, self.steps, self.done = state['stage'], state['steps'], state['done']
        self.rng.bit_generator.state = state['rng']

    def close(self):
        pass
