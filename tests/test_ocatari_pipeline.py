"""Real acceptance for steps 0/1, not yet a learning-pipeline integration test.

Select with `-m ocatari`; missing optional packages or ROMs are failures.
"""
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from bayesian_rrtl import ocatari_backend as compat
from bayesian_rrtl.games.pong import PONG_ACTIONS
from experiments.validate_ocatari_backend import check_terminal, check_truncation

pytestmark = [pytest.mark.atari, pytest.mark.ocatari]


def test_real_one_step_time_limit():
    assert check_truncation()["status"] == "passed"


def test_real_random_game_ends_naturally_and_frames_stay_immutable():
    result = check_terminal()
    assert result["terminated"] and not result["truncated"]
    assert result["steps"] < result["limit"]
    assert result["initial_frame_unchanged"]


@pytest.mark.parametrize("flags", [(False, False), (True, False), (False, True), (True, True)])
def test_installed_ocatari_step_with_simulated_underlying_env(monkeypatch, flags):
    # Exercise the installed OCAtari.step itself, not a double of its tuple ordering.
    compat.verified_versions()
    from ocatari.core import OCAtari
    from ocatari.ram.pong import _init_objects_ram
    oc = OCAtari.__new__(OCAtari)
    calls = []

    def step(action):
        calls.append(action)
        return None, 1, *flags, {}

    oc._env = SimpleNamespace(step=step, unwrapped=SimpleNamespace(get_action_meanings=lambda: PONG_ACTIONS))
    oc.objects = _init_objects_ram(hud=False)
    oc.obs_mode = "ori"
    oc.detect_objects = lambda: None
    oc._fill_buffer = lambda: None
    oc.close = lambda: None
    monkeypatch.setattr(compat, "_make_environment", lambda *args: oc)
    backend = compat.OCAtariBackend()
    backend.done = False  # Simulated underlying environment is already initialized.
    try:
        result = backend.step(3)
        assert (result.terminated, result.truncated) == flags
        assert result.raw_reward == 1
        assert calls == [3]
    finally:
        backend.close()


def test_real_single_emulator_step_no_hidden_reset_actions(monkeypatch):
    backend = compat.OCAtariBackend(max_episode_steps=20)
    try:
        underlying = backend.env._env
        calls = []
        original = underlying.step

        def counted(action):
            calls.append(action)
            return original(action)

        monkeypatch.setattr(underlying, "step", counted)
        frame = backend.reset(seed=41)
        saved = asdict(frame)
        assert calls == []
        assert not backend.env.create_rgb_stack
        assert not backend.env.create_dqn_stack
        assert not backend.env.create_ns_stack
        ale = underlying.unwrapped.ale
        before = ale.getFrameNumber()
        backend.step(2)
        assert calls == [2]
        assert ale.getFrameNumber() - before == 4
        assert asdict(frame) == saved
    finally:
        backend.close()
