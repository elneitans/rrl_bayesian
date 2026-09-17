"""Collection independent of Gym: an environment adapter supplies relational facts.

Contract: reset(seed) -> state; step(action) -> state, reward, raw_reward,
terminated, truncated. snapshot/restore must include the environment RNG,
wrapper counters, and any state needed by the observation extractor.
"""

import copy

from .checkpoint import load_checkpoint, save_checkpoint


class QTrainingSession:
    def __init__(self, agent, environment):
        self.agent = agent
        self.environment = environment
        self.state = None
        self.done = True
        self.episode_id = -1
        self.step_id = 0
        self.episode_return = self.raw_episode_return = 0.0
        self.history = []

    def step(self):
        if self.done:
            self.state = self.environment.reset(seed=self.agent.episode_seed())
            self.agent.action_buffer.clear()
            self.done = False
            self.episode_id += 1
            self.step_id = 0
            self.episode_return = self.raw_episode_return = 0.0
        epsilon = self.agent.epsilon
        action = self.agent.select_action(self.state)
        next_state, reward, raw_reward, terminated, truncated = self.environment.step(action)
        transition = self.agent.observe(self.state, action, reward, next_state, terminated, truncated,
                                        raw_reward=raw_reward, episode_id=self.episode_id, step_id=self.step_id)
        self.state = next_state
        self.done = terminated or truncated
        self.step_id += 1
        self.episode_return += reward
        self.raw_episode_return += raw_reward
        row = {'transition_id': transition.transition_id, 'interaction': self.agent.interactions,
               'episode_id': self.episode_id, 'step_id': self.step_id - 1, 'action': action,
               'reward': reward, 'raw_reward': raw_reward, 'terminated': terminated,
               'truncated': truncated, 'episode_return': self.episode_return,
               'raw_episode_return': self.raw_episode_return, 'epsilon': epsilon}
        self.history.append(row)
        self.agent.fit_if_due()
        return row

    def run(self, interactions):
        if type(interactions) is not int or interactions < 0:
            raise ValueError('Interaction budget must be a nonnegative integer')
        for _ in range(interactions):
            self.step()
        return self.history

    def save(self, path):
        collector = {key: copy.deepcopy(value) for key, value in vars(self).items()
                     if key not in ('agent', 'environment')}
        collector['environment'] = self.environment.snapshot()
        return save_checkpoint(self.agent, path, collector_state=collector)

    @classmethod
    def load(cls, path, environment, *, expected_config=None):
        agent, collector = load_checkpoint(path, expected_config=expected_config)
        if collector is None:
            raise ValueError('Checkpoint has no collector/environment state for exact resume')
        session = cls(agent, environment)
        if set(collector) != set(vars(session)) - {'agent'}:
            raise ValueError('Checkpoint collector fields mismatch')
        environment.restore(collector['environment'])
        vars(session).update({key: value for key, value in collector.items() if key != 'environment'})
        return session
