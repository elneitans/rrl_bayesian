"""OCAtari 2.2.1 compatibility boundary: one emulator, copied objects, Gym flags.

No import of OCAtari occurs until construction. No RGB extraction, hidden actions,
reward shaping or additional frame skipping occurs here. The pinned snapshot
codec stores emulator, wrapper and detector data without serializing live handles.
"""

import copy
from dataclasses import asdict
import hashlib
import importlib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import time
import math

import numpy as np

from .atari_backend import ObjectStep
from .game_registry import get_game_spec
from .objects import DetectedObject, ObjectFrame

VERIFIED_VERSIONS = {"ocatari": "2.2.1", "gymnasium": "1.1.1", "ale-py": "0.10.2"}
COMPATIBILITY_ID = "ocatari_2.2.1_gymnasium_1.1.1_ale_0.10.2_v1"


def verified_versions():
    try:
        installed = {name: version(name) for name in VERIFIED_VERSIONS}
    except PackageNotFoundError as exc:
        raise ImportError(
            "OCAtari backend needs optional dependencies in this interpreter: "
            "python -m pip install -r requirements-ocatari.txt"
        ) from exc
    if installed != VERIFIED_VERSIONS:
        raise ValueError(f"Unverified OCAtari compatibility: {installed}; expected {VERIFIED_VERSIONS}")
    return installed


def normalize_step_221(result):
    """2.2.1 source returns truncated BEFORE terminated; never infer from values."""
    _, reward, truncated, terminated, _ = result
    if not isinstance(terminated, (bool, np.bool_)) or not isinstance(truncated, (bool, np.bool_)):
        raise ValueError("OCAtari returned non-boolean episode flags")
    raw_reward = float(reward)
    if not math.isfinite(raw_reward):
        raise ValueError("OCAtari returned a nonfinite reward")
    return raw_reward, bool(terminated), bool(truncated)


def _make_environment(spec, max_episode_steps):
    from ocatari.core import OCAtari
    return OCAtari(
        spec.env_id, mode="ram", hud=False, obs_mode="ori", render_mode=None,
        create_buffer_stacks=[], frameskip=4, repeat_action_probability=0.0,
        full_action_space=False, max_episode_steps=max_episode_steps,
    )


class OCAtariBackend:
    backend_id = "ocatari_ram_v1"
    compatibility_id = COMPATIBILITY_ID

    def __init__(self, game="Pong", *, max_episode_steps=1000):
        self.spec = get_game_spec(game, self.backend_id)
        if self.spec.object_adapter is None:
            raise ValueError("Registered game has no object adapter")
        if type(max_episode_steps) is not int or max_episode_steps < 1:
            raise ValueError("Episode limit must be a positive integer")
        self.versions = verified_versions()
        self.env = _make_environment(self.spec, max_episode_steps)
        try:
            # OCAtari 2.2.1's public helper assumes two wrappers and fails with
            # Gymnasium's TimeLimit/OrderEnforcing/PassiveEnvChecker stack.
            self.actions = tuple(self.env._env.unwrapped.get_action_meanings())
            if self.actions != self.spec.actions:
                raise ValueError(f"Unexpected ALE action catalog: {self.actions}; expected {self.spec.actions}")
        except Exception:
            self.env.close()
            raise
        self.max_episode_steps = max_episode_steps
        self.done = True
        self.closed = False
        self.current_frame = None
        self.profile = {'backend_step_seconds': 0., 'backend_reset_seconds': 0.,
                        'object_conversion_seconds': 0.}

    def _frame(self):
        started = time.perf_counter()
        frame = self.spec.object_adapter(self.env.objects)
        self.profile['object_conversion_seconds'] += time.perf_counter() - started
        self.current_frame = frame
        return frame

    def reset(self, *, seed):
        if self.closed:
            raise ValueError("Backend is closed")
        self.done = True
        started = time.perf_counter()
        self.env.reset(seed=seed)
        self.profile['backend_reset_seconds'] += time.perf_counter() - started
        frame = self._frame()
        self.done = False
        return frame

    def step(self, action):
        if self.closed or self.done or type(action) is not int or not 0 <= action < len(self.actions):
            raise ValueError("Invalid action or completed episode")
        # Invalidate on conversion failure: the emulator has already advanced.
        self.done = True
        started = time.perf_counter()
        result = self.env.step(action)
        self.profile['backend_step_seconds'] += time.perf_counter() - started
        raw, terminated, truncated = normalize_step_221(result)
        frame = self._frame()
        self.done = terminated or truncated
        return ObjectStep(frame, raw, terminated, truncated)

    def manifest(self):
        from ale_py.roms import get_rom_path
        modules = ('ocatari.core', 'ocatari.ram.pong', 'ocatari.ram.game_objects',
                   'ocatari.ram.extract_ram_info', 'ale_py.env',
                   'bayesian_rrtl.ocatari_backend', 'bayesian_rrtl.games.pong')
        return {'backend': self.backend_id, 'compatibility_id': self.compatibility_id,
                'env_id': self.env._env.spec.id, 'versions': dict(self.versions),
                'source_hashes': {name: hashlib.sha256(Path(importlib.import_module(name).__file__).read_bytes()).hexdigest()
                                  for name in modules},
                'rom_sha256': hashlib.sha256(get_rom_path(self.spec.game.lower()).read_bytes()).hexdigest(),
                'actions': list(self.actions), 'mode': 'ram', 'hud': False, 'obs_mode': 'ori',
                'buffers': [], 'render_mode': None, 'reset_policy': 'none',
                'frameskip': 4, 'repeat_action_probability': 0., 'full_action_space': False,
                'max_episode_steps': self.max_episode_steps}

    def _wrappers(self):
        current, result = self.env._env, []
        while hasattr(current, 'env'):
            result.append(current)
            current = current.env
        return result

    @staticmethod
    def _rng_snapshot(owner):
        rng = getattr(owner, '_np_random', None)
        return {'state': None if rng is None else copy.deepcopy(rng.bit_generator.state),
                'seed': getattr(owner, '_np_random_seed', None)}

    @staticmethod
    def _restore_rng(owner, saved):
        owner._np_random = None
        if saved['state'] is not None:
            owner._np_random = np.random.default_rng()
            owner._np_random.bit_generator.state = saved['state']
        if hasattr(owner, '_np_random_seed'):
            owner._np_random_seed = saved['seed']

    def _check_no_buffers(self):
        if (self.env.create_rgb_stack or self.env.create_dqn_stack or self.env.create_ns_stack
                or any(getattr(self.env, name) is not None for name in
                       ('_state_buffer_rgb', '_state_buffer_dqn', '_state_buffer_ns'))):
            raise ValueError('This OCAtari snapshot codec requires disabled buffers')

    def detector_state(self):
        """Explicit copied detector data, also used by continuity diagnostics."""
        # GameObject and NoObject fields from the inspected 2.2.1 constructors.
        object_fields = ('_rgb', '_xy', 'wh', '_prev_xy', '_orientation', 'hud', '_visible')
        objects = []
        for obj in self.env.objects:
            names = object_fields + (('nslen',) if obj.category == 'NoObject' else ())
            if set(vars(obj)) != set(names):
                raise ValueError('Unrecognized mutable detector fields')
            objects.append({'category': obj.category,
                            'fields': {name: copy.deepcopy(getattr(obj, name)) for name in names}})
        return objects

    def snapshot(self):
        if self.closed:
            raise ValueError('Backend is closed')
        self._check_no_buffers()
        base = self.env._env.unwrapped
        fields = ('_elapsed_steps', '_max_episode_steps', '_has_reset', '_disable_render_order_enforcing',
                  'checked_reset', 'checked_step', 'checked_render', 'close_called')
        objects = self.detector_state()
        return {'kind': 'ocatari_ram_state_v1', 'manifest': self.manifest(),
                # ALE 0.10.2 serialize() decodes binary as UTF-8; its pickle codec
                # exposes the complete system state as a tuple of bytes instead.
                'ale_state': base.ale.cloneSystemState().__getstate__(),
                'rng': self._rng_snapshot(base),
                'action_rng': self._rng_snapshot(base.action_space),
                'observation_rng': self._rng_snapshot(base.observation_space),
                'wrappers': [(type(w).__name__, {key: getattr(w, key) for key in fields if key in vars(w)})
                             for w in self._wrappers()],
                'objects': objects, 'frame': None if self.current_frame is None else asdict(self.current_frame),
                'done': self.done, 'profile': copy.deepcopy(self.profile)}

    def validate_snapshot(self, snapshot):
        expected = {'kind', 'manifest', 'ale_state', 'rng', 'action_rng', 'observation_rng',
                    'wrappers', 'objects', 'frame', 'done', 'profile'}
        if not isinstance(snapshot, dict) or set(snapshot) != expected:
            raise ValueError('OCAtari snapshot fields mismatch')
        if snapshot['kind'] != 'ocatari_ram_state_v1' or snapshot['manifest'] != self.manifest():
            raise ValueError('OCAtari snapshot config, versions, source hashes or ROM mismatch')
        if [name for name, _ in snapshot['wrappers']] != [type(w).__name__ for w in self._wrappers()]:
            raise ValueError('OCAtari wrapper stack mismatch')
        fields = ('_elapsed_steps', '_max_episode_steps', '_has_reset', '_disable_render_order_enforcing',
                  'checked_reset', 'checked_step', 'checked_render', 'close_called')
        for wrapper, (_, saved) in zip(self._wrappers(), snapshot['wrappers']):
            if set(saved) != {key for key in fields if key in vars(wrapper)}:
                raise ValueError('OCAtari wrapper snapshot fields mismatch')
            for key in ('_max_episode_steps', '_disable_render_order_enforcing'):
                if key in saved and saved[key] != getattr(wrapper, key):
                    raise ValueError('OCAtari wrapper configuration mismatch')
        if type(snapshot['done']) is not bool:
            raise ValueError('OCAtari snapshot done must be boolean')
        self._check_no_buffers()

    def restore(self, snapshot):
        if self.closed:
            raise ValueError('Backend is closed')
        self.validate_snapshot(snapshot)
        from ale_py import ALEState
        from ocatari.ram.pong import Player, Ball, Enemy
        from ocatari.ram.game_objects import NoObject
        classes = {c.__name__: c for c in (Player, Ball, Enemy, NoObject)}
        restored = []
        for saved in snapshot['objects']:
            if saved['category'] not in classes:
                raise ValueError('Invalid Pong detector category')
            obj = classes[saved['category']]()
            if set(saved['fields']) != set(vars(obj)):
                raise ValueError('Invalid Pong detector snapshot fields')
            for key, value in saved['fields'].items():
                setattr(obj, key, copy.deepcopy(value))
            restored.append(obj)
        frame = snapshot['frame']
        if frame is not None:
            frame = ObjectFrame(tuple(DetectedObject(**obj) for obj in frame['objects']), frame['width'], frame['height'])
            if self.spec.object_adapter(restored) != frame:
                raise ValueError('Detector snapshot/frame mismatch')
        # Initialize Gym/RNG structures without an OCAtari detection or any action.
        self.env._env.reset(seed=0)
        base = self.env._env.unwrapped
        state = ALEState.__new__(ALEState)
        state.__setstate__(snapshot['ale_state'])
        base.ale.restoreSystemState(state)
        self._restore_rng(base, snapshot['rng'])
        self._restore_rng(base.action_space, snapshot['action_rng'])
        self._restore_rng(base.observation_space, snapshot['observation_rng'])
        for wrapper, (_, fields) in zip(self._wrappers(), snapshot['wrappers']):
            for key, value in fields.items():
                setattr(wrapper, key, value)
        self.env.objects = restored
        self.current_frame = frame
        self.done = snapshot['done']
        self.profile = copy.deepcopy(snapshot['profile'])

    def close(self):
        if not self.closed:
            self.env.close()
            self.closed = True
            self.done = True
