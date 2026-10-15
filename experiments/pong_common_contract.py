"""New common Pong contract; historical Atari registry and pipelines stay intact."""
import copy
import time
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path

import numpy as np

from bayesian_rrtl.games.pong import PONG_ACTIONS, PongRelationsV1
from bayesian_rrtl.objects import LegacyMultiGameExtractor
from experiments.pong_analysis_runtime import digest

CONTRACT = 'pong_common_v1'
PIPELINES = {'visual': 'pong_common_rgb_v1', 'ocatari': 'pong_common_ram_v1'}
WRAPPER_FIELDS = ('_elapsed_steps', '_max_episode_steps', '_has_reset',
                  '_disable_render_order_enforcing', 'checked_reset', 'checked_step',
                  'checked_render', 'close_called')


@dataclass(frozen=True)
class CommonPongConfig:
    pipeline: str
    rom_sha256: str
    horizon: int = 27000
    contract: str = CONTRACT

    def __post_init__(self):
        if self.pipeline not in PIPELINES or self.contract != CONTRACT:
            raise ValueError('Unknown common Pong pipeline/contract')
        if type(self.horizon) is not int or self.horizon < 1:
            raise ValueError('Invalid common Pong horizon')
        if len(self.rom_sha256) != 64 or any(c not in '0123456789abcdef' for c in self.rom_sha256):
            raise ValueError('Common Pong requires an explicit verified ROM SHA256')

    def manifest(self):
        return {**asdict(self), 'pipeline_id': PIPELINES[self.pipeline], 'env_id': 'ALE/Pong-v5',
                'frameskip': 4, 'repeat_action_probability': 0., 'actions': list(PONG_ACTIONS),
                'reset_policy': 'none', 'learning_reward': 'sign(raw_reward)',
                'raw_reward': 'separate', 'relations': PongRelationsV1().manifest()}


def installed_rom_hash():
    from ale_py.roms import get_rom_path
    from experiments.pong_analysis_common import sha
    return sha(get_rom_path('pong'))


class RGBDriver:
    """Exactly one Gym frameskip; reset executes no agent actions."""
    def __init__(self, horizon):
        import gymnasium as gym
        import ale_py
        from bayesian_rrtl.ocatari_backend import VERIFIED_VERSIONS
        expected = {name: VERIFIED_VERSIONS[name] for name in ('gymnasium', 'ale-py')}
        if {name: version(name) for name in expected} != expected:
            raise ValueError('Common RGB requires the same verified Gym/ALE runtime as RAM')
        gym.register_envs(ale_py)
        self.env = gym.make('ALE/Pong-v5', obs_type='rgb', frameskip=4,
                            repeat_action_probability=0., full_action_space=False,
                            max_episode_steps=horizon)
        self.actions = tuple(self.env.unwrapped.get_action_meanings())
        self.extractor = LegacyMultiGameExtractor('Pong')
        self.current_frame = None

    def reset(self, *, seed):
        self.extractor.reset()
        rgb, _ = self.env.reset(seed=seed)
        self.current_frame = self.extractor.extract(rgb)
        return self.current_frame

    def step(self, action):
        from bayesian_rrtl.atari_backend import ObjectStep
        rgb, raw, terminal, truncated, _ = self.env.step(action)
        self.current_frame = self.extractor.extract(rgb)
        return ObjectStep(self.current_frame, float(raw), bool(terminal), bool(truncated))

    def wrappers(self):
        result, current = [], self.env
        while hasattr(current, 'env'):
            result.append(current)
            current = current.env
        return result

    def snapshot(self):
        base = self.env.unwrapped
        return {'ale': base.ale.cloneSystemState().__getstate__(),
                'rng': copy.deepcopy(base.np_random.bit_generator.state),
                'wrappers': [(type(w).__name__, {k: copy.deepcopy(vars(w)[k])
                             for k in WRAPPER_FIELDS if k in vars(w)}) for w in self.wrappers()],
                'extractor': self.extractor.snapshot(), 'frame': self.current_frame}

    def restore(self, saved):
        from ale_py import ALEState
        if [n for n, _ in saved['wrappers']] != [type(w).__name__ for w in self.wrappers()]:
            raise ValueError('Common Pong wrapper mismatch')
        self.env.reset(seed=0)
        state = ALEState.__new__(ALEState)
        state.__setstate__(saved['ale'])
        self.env.unwrapped.ale.restoreSystemState(state)
        self.env.unwrapped.np_random.bit_generator.state = saved['rng']
        for wrapper, (_, fields) in zip(self.wrappers(), saved['wrappers']):
            vars(wrapper).update(fields)
        self.extractor.restore(saved['extractor'])
        self.current_frame = saved['frame']

    def close(self):
        self.env.close()


class CommonPongEnvironment:
    actions = PONG_ACTIONS

    def __init__(self, config, *, driver=None):
        self.config = config
        self.driver_provenance = 'verified_atari' if driver is None else 'injected_driver'
        if driver is None:
            if installed_rom_hash() != config.rom_sha256:
                raise ValueError('Common Pong ROM checksum mismatch')
            if config.pipeline == 'visual':
                driver = RGBDriver(config.horizon)
            else:
                from bayesian_rrtl.ocatari_backend import OCAtariBackend
                driver = OCAtariBackend('Pong', max_episode_steps=config.horizon)
        if tuple(driver.actions) != self.actions:
            driver.close()
            raise ValueError('Common Pong action order mismatch')
        self.backend = driver
        self.relations = PongRelationsV1()
        self.done = True
        self.steps = 0
        self.last_auxiliary = None

    def encoder(self):
        return self.relations.encoder()

    def _facts(self, frame, raw=0.):
        started = time.perf_counter()
        had_history = self.relations.previous is not None and raw == 0
        previous = ({o.identity: o for o in self.relations.previous.objects} if had_history else {})
        facts = self.relations.transform(frame, raw_reward=raw)
        objects = {o.identity: o for o in frame.objects}
        causes = []
        observed = {(f.type, f.name, f.obj1, f.obj2) for f in facts}
        for key in self.encoder().catalog:
            if (key.type, key.name, key.obj1, key.obj2) in observed:
                causes.append('observed')
            elif any(not objects[n.split('_t')[0]].present
                     for n in (key.obj1, key.obj2) if n and '_t-1' not in n):
                causes.append('observed_absence')
            elif not had_history and any(n and '_t-1' in n for n in (key.obj1, key.obj2)):
                causes.append('history_reset')
            elif any(n and '_t-1' in n and not previous[n.split('_t')[0]].present
                     for n in (key.obj1, key.obj2)):
                causes.append('observed_absence')
            else:
                causes.append('relation_discarded')
        self.last_auxiliary = {'objects': asdict(frame), 'relation_causes': causes,
                               'history_reset': not had_history}
        self.last_relation_seconds = time.perf_counter()-started
        return facts

    def reset(self, *, seed):
        self.relations.reset()
        self.steps = 0
        frame = self.backend.reset(seed=seed)
        self.done = False
        return self._facts(frame)

    def step(self, action):
        if self.done or type(action) is not int or not 0 <= action < len(self.actions):
            raise ValueError('Invalid common Pong action or completed episode')
        result = self.backend.step(action)
        self.steps += 1
        terminal = bool(result.terminated)
        truncated = bool(result.truncated or (self.steps >= self.config.horizon and not terminal))
        self.done = terminal or truncated
        return self._facts(result.frame, result.raw_reward), float(np.sign(result.raw_reward)), float(result.raw_reward), terminal, truncated

    def runtime_manifest(self):
        from experiments.pong_analysis_common import sha
        return {'contract': self.config.manifest(), 'driver_provenance': self.driver_provenance,
                'runtime': {name: version(name) for name in ('gymnasium', 'ale-py')},
                'code_sha256': sha(Path(__file__)),
                'backend': self.backend.manifest() if hasattr(self.backend, 'manifest') else PIPELINES[self.config.pipeline]}

    def snapshot(self):
        return {'manifest': self.runtime_manifest(), 'environment': self.backend.snapshot(),
                'relations': self.relations.snapshot(), 'steps': self.steps, 'done': self.done,
                'auxiliary': copy.deepcopy(self.last_auxiliary)}

    def restore(self, saved):
        if saved['manifest'] != self.runtime_manifest():
            raise ValueError('Common Pong snapshot contract/runtime mismatch')
        self.backend.restore(saved['environment'])
        self.relations.restore(saved['relations'])
        self.steps, self.done = saved['steps'], saved['done']
        self.last_auxiliary = copy.deepcopy(saved['auxiliary'])

    def close(self):
        self.backend.close()


@dataclass(frozen=True)
class CommonPongFactory:
    config: CommonPongConfig
    def __call__(self):
        return CommonPongEnvironment(self.config)
    def with_horizon(self, horizon):
        return CommonPongFactory(CommonPongConfig(self.config.pipeline, self.config.rom_sha256, horizon))


def validate_contract(expected, actual):
    if digest(expected) != digest(actual):
        raise ValueError('Pong protocol contract mismatch; historical states cannot be relabelled')
