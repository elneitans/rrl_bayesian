"""Backend contract with doubles: no OCAtari, emulator or ROM required."""

from dataclasses import FrozenInstanceError, replace
from importlib.metadata import PackageNotFoundError
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest

from bayesian_rrtl.atari_backend import ObjectBackend
from bayesian_rrtl.game_registry import GameRegistry, get_game_spec
from bayesian_rrtl.games.pong import PONG_ACTIONS, PONG_OCATARI, pong_object_frame
from bayesian_rrtl import ocatari_backend as compat


class Obj:
    def __init__(self, category, bbox=(0, 0, 4, 15), present=True):
        self.category, self.xywh, self.present = category, list(bbox), present

    def __bool__(self):
        return self.present


def objects():
    return [Obj("Player"), Obj("Ball", (10, 20, 2, 4)), Obj("Enemy")]


class FakeGym:
    def __init__(self, flags):
        self.calls = []
        self.flags = flags

    def step(self, action):
        self.calls.append(action)
        terminated, truncated = self.flags
        return None, -1, terminated, truncated, {}


class FakeOC:
    def __init__(self, flags=(False, False)):
        self.objects = objects()
        self.underlying = FakeGym(flags)
        self._env = SimpleNamespace(unwrapped=self)
        self.resets = []
        self.closed = False

    def get_action_meanings(self):
        return PONG_ACTIONS

    def reset(self, *, seed):
        self.resets.append(seed)
        self.objects = objects()
        return None, {}

    def step(self, action):
        obs, reward, terminated, truncated, info = self.underlying.step(action)
        self.objects[0].xywh[1] += 3
        self.objects[1] = Obj("NoObject", present=False)
        return obs, reward, truncated, terminated, info

    def close(self):
        self.closed = True


@pytest.fixture
def setup_backend(monkeypatch):
    monkeypatch.setattr(compat, "version", compat.VERIFIED_VERSIONS.__getitem__)

    def setup(fake):
        monkeypatch.setattr(compat, "_make_environment", lambda *args: fake)
        return compat.OCAtariBackend()
    return setup


@pytest.mark.parametrize("flags", [(False, False), (True, False), (False, True), (True, True)])
def test_flags_single_step_and_final_objects(setup_backend, flags):
    fake = FakeOC(flags)
    backend = setup_backend(fake)
    assert isinstance(backend, ObjectBackend)
    before = backend.reset(seed=41)
    original = repr(before)
    assert fake.resets == [41] and fake.underlying.calls == []
    step = backend.step(5)
    assert fake.underlying.calls == [5]
    assert (step.terminated, step.truncated) == flags
    assert step.raw_reward == -1.0
    assert not next(o for o in step.frame.objects if o.identity == "ball").present
    assert repr(before) == original
    assert next(o for o in step.frame.objects if o.identity == "player").bbox[1] == 3.0
    with pytest.raises(FrozenInstanceError):
        step.terminated = True
    if any(flags):
        with pytest.raises(ValueError, match="completed"):
            backend.step(0)
        assert fake.underlying.calls == [5]
    backend.close()
    assert fake.closed


def test_slot_absence_border_and_copied_coordinates():
    source = objects()
    source[0].xywh = [-2, 0, 4, 15]
    source[1] = Obj("NoObject", present=False)
    frame = pong_object_frame(source)
    slots = {o.identity: o for o in frame.objects}
    assert (frame.width, frame.height) == (160, 210)
    assert set(slots) == {"player", "ball", "enemy"}
    assert slots["player"].bbox == (-2.0, 0.0, 4.0, 15.0)
    assert all(type(v) is float for v in slots["player"].bbox)
    assert slots["enemy"].present and slots["enemy"].bbox[:2] == (0.0, 0.0)
    assert slots["ball"].bbox is None and not slots["ball"].present
    source[0].xywh[0] = 999
    assert slots["player"].bbox[0] == -2.0


@pytest.mark.parametrize("bbox", [(0, 0, 0, 4), (0, 0, 2, -1),
                                  (float("nan"), 0, 2, 4), (0, float("inf"), 2, 4),
                                  (0, 0, 2), (0, 0, "bad", 4), None])
def test_invalid_bbox(bbox):
    source = objects()
    source[1].xywh = bbox
    with pytest.raises(ValueError, match="bbox.*ball"):
        pong_object_frame(source)


@pytest.mark.parametrize("change", ["extra", "missing", "swapped", "duplicate"])
def test_invalid_slot_layout(change):
    source = objects()
    if change == "extra":
        source.append(Obj("PlayerScore"))
    elif change == "missing":
        source.pop()
    elif change == "swapped":
        source[0], source[1] = source[1], source[0]
    else:
        source[1] = source[0]
    with pytest.raises(ValueError, match="slots|category"):
        pong_object_frame(source)


@pytest.mark.parametrize("package", compat.VERIFIED_VERSIONS)
def test_unverified_versions_rejected_before_creation(monkeypatch, package):
    versions = {**compat.VERIFIED_VERSIONS, package: "99.0"}
    monkeypatch.setattr(compat, "version", versions.__getitem__)
    monkeypatch.setattr(compat, "_make_environment", lambda *args: pytest.fail("created environment"))
    with pytest.raises(ValueError, match="Unverified"):
        compat.OCAtariBackend()


def test_missing_dependency_has_installation_hint(monkeypatch):
    def missing(name):
        raise PackageNotFoundError(name)
    monkeypatch.setattr(compat, "version", missing)
    with pytest.raises(ImportError, match="requirements-ocatari.txt"):
        compat.OCAtariBackend()


def test_imports_without_optional_dependency():
    code = '''
import sys
from importlib.abc import MetaPathFinder
class NoOCAtari(MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] == 'ocatari':
            raise ImportError('deliberately unavailable')
sys.meta_path.insert(0, NoOCAtari())
import bayesian_rrtl.agent
import bayesian_rrtl.atari
import bayesian_rrtl.ocatari_backend
from bayesian_rrtl.game_registry import get_game_spec
assert get_game_spec('Pong', 'legacy_visual_pong_v1').exact_resume
assert not any(k.startswith('ocatari') for k in sys.modules)
'''
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_action_mismatch_closes_before_reset(setup_backend):
    fake = FakeOC()
    fake.get_action_meanings = lambda: tuple(reversed(PONG_ACTIONS))
    with pytest.raises(ValueError, match="action catalog"):
        setup_backend(fake)
    assert fake.closed and not fake.resets and not fake.underlying.calls


@pytest.mark.parametrize("action", [-1, 6, True, 1.5])
def test_invalid_actions_do_not_advance(setup_backend, action):
    fake = FakeOC()
    backend = setup_backend(fake)
    backend.reset(seed=41)
    with pytest.raises(ValueError):
        backend.step(action)
    assert not fake.underlying.calls
    backend.close()


def test_reset_required_closed_and_snapshot_capability(setup_backend):
    fake = FakeOC()
    backend = setup_backend(fake)
    with pytest.raises(ValueError):
        backend.step(0)
    assert backend.spec.exact_resume and backend.spec.snapshot_version == "atari_relational_v2"
    backend.close()
    with pytest.raises(ValueError, match="closed"):
        backend.reset(seed=41)


def test_constructor_arguments_and_fresh_buffer_list(monkeypatch):
    made = []

    def constructor(*args, **kwargs):
        made.append((args, kwargs))
        return FakeOC()

    core = ModuleType("ocatari.core")
    core.OCAtari = constructor
    monkeypatch.setitem(sys.modules, "ocatari", ModuleType("ocatari"))
    monkeypatch.setitem(sys.modules, "ocatari.core", core)
    monkeypatch.setattr(compat, "version", compat.VERIFIED_VERSIONS.__getitem__)
    a = compat.OCAtariBackend(max_episode_steps=17)
    b = compat.OCAtariBackend(max_episode_steps=17)
    assert made[0][0] == ("ALE/Pong-v5",)
    assert made[0][1] == dict(mode="ram", hud=False, obs_mode="ori", render_mode=None,
                             create_buffer_stacks=[], frameskip=4, repeat_action_probability=0.0,
                             full_action_space=False, max_episode_steps=17)
    assert made[0][1]["create_buffer_stacks"] is not made[1][1]["create_buffer_stacks"]
    a.close()
    b.close()


def test_registry_capabilities_and_explicit_extension(monkeypatch):
    assert get_game_spec("Pong", "ocatari_ram_v1") == PONG_OCATARI
    assert PONG_OCATARI.representations == ("comparative",)
    for game in ("Breakout", "Pong", "DemonAttack"):
        assert get_game_spec(game, f"legacy_visual_{game.lower()}_v1").relation_factory
    with pytest.raises(ValueError, match="Unsupported"):
        compat.OCAtariBackend("Breakout")
    registry = GameRegistry()
    fake = replace(PONG_OCATARI, game="Fictional", env_id="Fake-v0",
                   object_adapter=lambda source: pong_object_frame(source))
    registry.register(fake)
    assert registry.get("Fictional", "ocatari_ram_v1") is fake
    with pytest.raises(ValueError, match="already registered"):
        registry.register(fake)
    with pytest.raises(ValueError, match="Unsupported"):
        registry.get("Unknown", "ocatari_ram_v1")
    monkeypatch.setattr(compat, "get_game_spec", registry.get)
    monkeypatch.setattr(compat, "version", compat.VERIFIED_VERSIONS.__getitem__)
    monkeypatch.setattr(compat, "_make_environment", lambda *args: FakeOC())
    backend = compat.OCAtariBackend("Fictional")
    assert backend.reset(seed=41) == pong_object_frame(objects())
    backend.close()
