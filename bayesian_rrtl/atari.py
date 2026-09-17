"""Breakout adapter with replaceable object extraction and ALE system snapshots."""

import copy
from dataclasses import asdict, dataclass
from importlib.metadata import version
import time

import numpy as np

from .objects import BreakoutRelations, MultiGameRelations, make_extractor


@dataclass(frozen=True)
class AtariConfig:
    game: str = 'Breakout'
    extractor: str = 'legacy_visual_breakout_v1'
    representation: str = 'comparative'
    include_incomplete_states: bool = False
    frameskip: int = 4
    repeat_action_probability: float = 0.0
    max_episode_steps: int = 1000

    def __post_init__(self):
        if self.game not in ('Breakout', 'Pong', 'DemonAttack'):
            raise ValueError('Unsupported Atari game')
        if self.extractor != f'legacy_visual_{self.game.lower()}_v1':
            raise ValueError('Only legacy_visual_breakout_v1 is implemented; OCAtari is deferred')
        if self.representation not in ('comparative', 'logical'):
            raise ValueError('Invalid representation')
        if any(type(v) is not int or v < 1 for v in (self.frameskip, self.max_episode_steps)):
            raise ValueError('Frame skip and episode limit must be positive integers')
        if not 0 <= self.repeat_action_probability <= 1 or type(self.include_incomplete_states) is not bool:
            raise ValueError('Invalid Atari parameters')

    def to_dict(self):
        return asdict(self)


class AtariRelationalEnvironment:
    actions = ('NOOP', 'FIRE', 'RIGHT', 'LEFT')

    def __init__(self, config=AtariConfig(), *, extractor=None):
        import gymnasium as gym
        import ale_py
        from .baseline import FireReset
        gym.register_envs(ale_py)
        self.config = config
        self.actions = game_actions(config.game)
        self.extractor = extractor or make_extractor(config.extractor)
        if self.extractor.backend_id != config.extractor:
            raise ValueError('Extractor identity differs from environment configuration')
        self.relations = (BreakoutRelations(config.representation, config.include_incomplete_states)
                          if config.game == 'Breakout' else
                          MultiGameRelations(config.game, config.representation, config.include_incomplete_states))
        self.env = FireReset(gym.make(f'{config.game}NoFrameskip-v4', obs_type='rgb',
            frameskip=config.frameskip, repeat_action_probability=config.repeat_action_probability,
            full_action_space=False, max_episode_steps=config.max_episode_steps))
        if tuple(self.env.unwrapped.get_action_meanings()) != self.actions:
            self.env.close()
            raise ValueError('Unexpected ALE action catalog')
        self.done = True
        self.profile = {'extract_seconds': 0.0, 'relations_seconds': 0.0, 'environment_seconds': 0.0,
                        'frames_extracted': 0, 'empty_states': 0}

    def encoder(self):
        return self.relations.encoder()

    def _state(self, observation):
        started = time.perf_counter()
        objects = self.extractor.extract(observation)
        self.profile['extract_seconds'] += time.perf_counter() - started
        started = time.perf_counter()
        state = self.relations.transform(objects)
        self.profile['relations_seconds'] += time.perf_counter() - started
        self.profile['frames_extracted'] += 1
        self.profile['empty_states'] += int(not state)
        return state

    def reset(self, *, seed):
        self.extractor.reset()
        self.relations.reset()
        started = time.perf_counter()
        observation, _ = self.env.reset(seed=seed)
        self.profile['environment_seconds'] += time.perf_counter() - started
        self.done = False
        return self._state(observation)

    def step(self, action):
        if self.done or type(action) is not int or not 0 <= action < len(self.actions):
            raise ValueError('Invalid action or completed episode')
        started = time.perf_counter()
        observation, raw, terminated, truncated, _ = self.env.step(action)
        self.profile['environment_seconds'] += time.perf_counter() - started
        self.done = bool(terminated or truncated)
        return self._state(observation), float(np.sign(raw)) + (0.1 if self.config.game == 'Pong' else 0.0), float(raw), bool(terminated), bool(truncated)

    def _wrappers(self):
        current, result = self.env, []
        while hasattr(current, 'env'):
            result.append(current)
            current = current.env
        return result

    def snapshot(self):
        fields = ('_elapsed_steps', '_has_reset', 'checked_reset', 'checked_step', 'checked_render')
        return {'kind': 'atari_relational_v1', 'config': self.config.to_dict(),
                'versions': {name: version(name) for name in ('gymnasium', 'ale-py')},
                'ale': self.env.unwrapped.ale.cloneSystemState(),
                'numpy_rng': copy.deepcopy(self.env.unwrapped.np_random.bit_generator.state),
                'wrappers': [(type(w).__name__, {key: vars(w)[key] for key in fields if key in vars(w)})
                             for w in self._wrappers()],
                'extractor': self.extractor.snapshot(), 'previous': copy.deepcopy(self.relations.previous),
                'done': self.done, 'profile': copy.deepcopy(self.profile)}

    def restore(self, snapshot):
        if snapshot['kind'] != 'atari_relational_v1' or snapshot['config'] != self.config.to_dict():
            raise ValueError('Atari snapshot config mismatch')
        if snapshot['versions'] != {name: version(name) for name in ('gymnasium', 'ale-py')}:
            raise ValueError('Exact Atari resume requires matching Gymnasium/ALE versions')
        wrappers = self._wrappers()
        if [type(w).__name__ for w in wrappers] != [name for name, _ in snapshot['wrappers']]:
            raise ValueError('Atari wrapper stack mismatch')
        self.env.reset(seed=0)
        self.env.unwrapped.ale.restoreSystemState(snapshot['ale'])
        self.env.unwrapped.np_random.bit_generator.state = snapshot['numpy_rng']
        for wrapper, (_, fields) in zip(wrappers, snapshot['wrappers']):
            vars(wrapper).update(fields)
        self.extractor.restore(snapshot['extractor'])
        self.relations.previous = copy.deepcopy(snapshot['previous'])
        self.done = snapshot['done']
        self.profile = copy.deepcopy(snapshot['profile'])

    def close(self):
        self.env.close()


def game_actions(game):
    if game == 'Breakout':
        return ('NOOP', 'FIRE', 'RIGHT', 'LEFT')
    if game in ('Pong', 'DemonAttack'):
        return ('NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE')
    raise ValueError('Unknown game')
