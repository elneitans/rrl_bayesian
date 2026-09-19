"""Breakout adapter with replaceable object extraction and ALE system snapshots."""

import copy
from dataclasses import asdict, dataclass
from importlib.metadata import version
import time

import numpy as np

from .objects import make_extractor
from .game_registry import get_game_spec


@dataclass(frozen=True)
class AtariConfig:
    game: str = 'Breakout'
    extractor: str = 'legacy_visual_breakout_v1'
    representation: str = 'comparative'
    include_incomplete_states: bool = False
    frameskip: int = 4
    repeat_action_probability: float = 0.0
    max_episode_steps: int = 1000

    env_id: str | None = None
    relation_schema: str | None = None
    reward_transform: str | None = None
    reset_policy: str | None = None

    def __post_init__(self):
        try:
            spec = get_game_spec(self.game, self.extractor)
        except ValueError as exc:
            raise ValueError(f'Unsupported Atari/OCAtari game/backend: {self.game}/{self.extractor}') from exc
        if self.representation not in spec.representations:
            raise ValueError('Invalid representation for backend')
        if self.include_incomplete_states not in spec.incomplete_states:
            raise ValueError('Unsupported incomplete-state option for backend')
        resolved = self.resolved_dict()
        if (resolved['env_id'] != spec.env_id or resolved['relation_schema'] != spec.relation_schema
                or resolved['reset_policy'] not in spec.reset_policies
                or resolved['reward_transform'] not in spec.reward_transforms):
            raise ValueError('Environment ID, relations, reset or reward differs from backend contract')
        if spec.backend_factory is not None and (self.frameskip != 4 or self.repeat_action_probability != 0):
            raise ValueError('OCAtari RAM requires frameskip=4 and repeat_action_probability=0')
        if any(type(v) is not int or v < 1 for v in (self.frameskip, self.max_episode_steps)):
            raise ValueError('Frame skip and episode limit must be positive integers')
        if not 0 <= self.repeat_action_probability <= 1 or type(self.include_incomplete_states) is not bool:
            raise ValueError('Invalid Atari parameters')

    def to_dict(self):
        # Preserve the historical serialized shape when additive fields were absent.
        return {key: value for key, value in asdict(self).items() if value is not None}

    def resolved_dict(self):
        spec = get_game_spec(self.game, self.extractor)
        result = asdict(self)
        for key, default in (('env_id', spec.env_id), ('relation_schema', spec.relation_schema),
                             ('reward_transform', spec.reward_transforms[0]), ('reset_policy', spec.reset_policies[0])):
            if result[key] is None:
                result[key] = default
        return result

    @property
    def reward_bounds(self):
        return (-.9, 1.1) if self.resolved_dict()['reward_transform'] == 'sign_plus_0.1' else (-1., 1.)


class AtariRelationalEnvironment:
    actions = ('NOOP', 'FIRE', 'RIGHT', 'LEFT')

    def __init__(self, config=AtariConfig(), *, extractor=None):
        self.config = config
        self.spec = get_game_spec(config.game, config.extractor)
        self.actions = self.spec.actions
        self.relations = self.spec.relation_factory(config.representation, config.include_incomplete_states)
        self.backend = None
        self.done = True
        if self.spec.backend_factory is not None:
            if extractor is not None:
                raise ValueError('Object backends do not accept a visual extractor')
            self.backend = self.spec.backend_factory(max_episode_steps=config.max_episode_steps)
            self.env = self.backend
            self.actions = self.backend.actions
            if self.actions != self.spec.actions:
                self.backend.close()
                raise ValueError('Unexpected backend action catalog')
            self.profile = {'backend_step_seconds': 0., 'backend_reset_seconds': 0.,
                            'object_conversion_seconds': 0., 'relations_seconds': 0.,
                            'frames_extracted': 0, 'empty_states': 0}
            return
        import gymnasium as gym
        import ale_py
        from .baseline import FireReset
        gym.register_envs(ale_py)
        self.extractor = extractor or make_extractor(config.extractor)
        if self.extractor.backend_id != config.extractor:
            raise ValueError('Extractor identity differs from environment configuration')
        self.env = FireReset(gym.make(config.resolved_dict()['env_id'], obs_type='rgb',
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

    def _object_state(self, frame, raw_reward=0.):
        started = time.perf_counter()
        state = self.relations.transform(frame, raw_reward=raw_reward)
        self.profile['relations_seconds'] += time.perf_counter() - started
        self.profile.update(self.backend.profile)
        self.profile['frames_extracted'] += 1
        self.profile['empty_states'] += int(not state)
        return state

    def reset(self, *, seed):
        if self.backend is not None:
            self.relations.reset()
            frame = self.backend.reset(seed=seed)
            self.done = False
            return self._object_state(frame)
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
        if self.backend is not None:
            step = self.backend.step(action)
            self.done = step.terminated or step.truncated
            return (self._object_state(step.frame, step.raw_reward), self._reward(step.raw_reward),
                    step.raw_reward, step.terminated, step.truncated)
        started = time.perf_counter()
        observation, raw, terminated, truncated, _ = self.env.step(action)
        self.profile['environment_seconds'] += time.perf_counter() - started
        self.done = bool(terminated or truncated)
        return self._state(observation), self._reward(raw), float(raw), bool(terminated), bool(truncated)

    def _reward(self, raw):
        bonus = 0.1 if self.config.resolved_dict()['reward_transform'] == 'sign_plus_0.1' else 0.0
        return float(np.sign(raw)) + bonus

    def runtime_manifest(self):
        if self.backend is None:
            return {'backend': self.config.extractor, 'env_id': self.env.unwrapped.spec.id,
                    'actions': list(self.actions), 'catalog': self.encoder().to_dict()}
        return {**self.backend.manifest(), 'relations': self.relations.manifest(),
                'catalog': self.encoder().to_dict()}

    def validate_snapshot(self, snapshot):
        if AtariConfig(**snapshot['config']).resolved_dict() != self.config.resolved_dict():
            raise ValueError('Atari snapshot config mismatch')
        expected = 'atari_relational_v2' if self.backend is not None else 'atari_relational_v1'
        if snapshot['kind'] != expected:
            raise ValueError('Atari snapshot version/backend mismatch')
        if self.backend is not None:
            self.backend.validate_snapshot(snapshot['backend'])
            if snapshot['done'] != snapshot['backend']['done']:
                raise ValueError('Atari/backend snapshot done mismatch')
            if snapshot['relations']['previous'] != snapshot['backend']['frame']:
                raise ValueError('Atari relation history/current frame mismatch')
            # Also reject changed semantics for evaluation, not just resumption.
            if snapshot['relations']['manifest'] != self.relations.manifest():
                raise ValueError('Pong relation snapshot semantics mismatch')

    def _wrappers(self):
        current, result = self.env, []
        while hasattr(current, 'env'):
            result.append(current)
            current = current.env
        return result

    def snapshot(self):
        if self.backend is not None:
            return {'kind': 'atari_relational_v2', 'config': self.config.resolved_dict(),
                    'backend': self.backend.snapshot(), 'relations': self.relations.snapshot(),
                    'done': self.done, 'profile': copy.deepcopy(self.profile)}
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
        self.validate_snapshot(snapshot)
        if self.backend is not None:
            self.backend.restore(snapshot['backend'])
            self.relations.restore(snapshot['relations'])
            self.done = snapshot['done']
            self.profile = copy.deepcopy(snapshot['profile'])
            return
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
