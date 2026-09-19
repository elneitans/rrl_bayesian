"""Facade/runner integration without changes to H5 learning algorithms."""
from dataclasses import replace
import json
from pathlib import Path
import pickle
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from bayesian_rrtl.agent import BayesianQAgent
from bayesian_rrtl.atari import AtariConfig, AtariRelationalEnvironment
from bayesian_rrtl.atari_backend import ObjectStep
from bayesian_rrtl.checkpoint import load_checkpoint
from bayesian_rrtl.game_registry import GameRegistry, get_game_spec
from bayesian_rrtl.objects import DetectedObject, ObjectFrame
from bayesian_rrtl.q_config import BayesianQConfig
from bayesian_rrtl.targets import frozen_q_targets
from bayesian_rrtl.training import QTrainingSession
from train_atari import validate_pair, run_corrected


def config():
    return AtariConfig(game='Pong', extractor='ocatari_ram_v1', include_incomplete_states=True,
                       max_episode_steps=120)


def agent_config():
    return BayesianQConfig(actions=get_game_spec('Pong', 'ocatari_ram_v1').actions,
                           seed=41, gamma=.99, reward_min=-1., reward_max=1.,
                           m=1, burn_in=2, draws=4, fit_interval=8, batch_size_per_action=8,
                           validation_seeds=(310001,), test_seeds=(410001,))


def test_legacy_config_serialization_and_resolved_defaults():
    old = dict(game='Pong', extractor='legacy_visual_pong_v1', representation='comparative',
               include_incomplete_states=False, frameskip=4, repeat_action_probability=0., max_episode_steps=1000)
    c = AtariConfig(**old)
    assert c.to_dict() == old
    assert c.resolved_dict()['reward_transform'] == 'sign_plus_0.1'
    assert c.resolved_dict()['env_id'] == 'PongNoFrameskip-v4'
    assert c.resolved_dict()['reset_policy'] == 'legacy_fire'
    assert c.reward_bounds == (-.9, 1.1)
    assert AtariConfig(**c.resolved_dict()).resolved_dict() == c.resolved_dict()
    assert config().reward_bounds == (-1., 1.)


@pytest.mark.parametrize('changes', [dict(game='Breakout'), dict(representation='logical'),
    dict(include_incomplete_states=False), dict(frameskip=1), dict(repeat_action_probability=.25),
    dict(reward_transform='sign_plus_0.1'), dict(env_id='PongNoFrameskip-v4'),
    dict(relation_schema='legacy_pong_v1'), dict(reset_policy='legacy_fire')])
def test_unsupported_configuration_fails_early(changes):
    with pytest.raises(ValueError):
        replace(config(), **changes)


def test_reward_bounds_actual_actions_and_corrected_validation(tmp_path):
    validate_pair(config(), agent_config())
    with pytest.raises(ValueError, match='Reward bounds'):
        validate_pair(config(), replace(agent_config(), reward_min=-.9, reward_max=1.1))
    with pytest.raises(ValueError, match='actions'):
        validate_pair(config(), agent_config(), SimpleNamespace(actions=('NOOP',)))
    with pytest.raises(ValueError, match='corrected'):
        run_corrected(config(), agent_config(), tmp_path / 'absent', 8, 'none')
    assert not (tmp_path / 'absent').exists()


def test_facade_uses_registry_and_keeps_points_flags_and_bootstrap(monkeypatch):
    # Fictitious backend tests registry extension through the facade itself.
    import bayesian_rrtl.atari as facade
    frame = ObjectFrame(tuple(DetectedObject(name, True, (20., 30., 4., 12.))
                              for name in ('player', 'ball', 'enemy')))
    expected = get_game_spec('Pong', 'ocatari_ram_v1')
    class Backend:
        actions = expected.actions
        profile = {'backend_step_seconds': 0., 'backend_reset_seconds': 0., 'object_conversion_seconds': 0.}
        def __init__(self):
            self.steps = 0
        def reset(self, *, seed):
            return frame
        def step(self, action):
            self.steps += 1
            return ObjectStep(frame, (2., -3., 0.)[self.steps - 1], self.steps == 3, self.steps == 2)
        def close(self):
            pass
    backend = Backend()
    registry = GameRegistry()
    registry.register(replace(expected, game='Fictional', env_id='Fake-v0',
                              backend_factory=lambda **kwargs: backend))
    base_config = config()
    monkeypatch.setattr(facade, 'get_game_spec', registry.get)
    env = AtariRelationalEnvironment(replace(base_config, game='Fictional'))
    try:
        session = QTrainingSession(BayesianQAgent(agent_config(), env.encoder()), env)
        session.run(1)
        assert not session.done and backend.steps == 1
        assert session.history[0]['reward'] == 1. and session.history[0]['raw_reward'] == 2.
        assert all(f.value is False for f in session.state if f.name == 'present' and f.obj1.endswith('t-1'))
        session.run(2)
        transitions = session.agent.replay.transitions
        assert [t.terminated for t in transitions] == [False, False, True]
        assert [t.truncated for t in transitions] == [False, True, False]
        assert transitions[1].next_state is not None
        targets = frozen_q_targets(transitions, SimpleNamespace(predict=lambda batch: np.full((len(batch), 6), 10.)),
                                   env.encoder(), gamma=.99)
        np.testing.assert_allclose(targets, [10.9, 8.9, 0.])
    finally:
        env.close()


@pytest.mark.ocatari
@pytest.mark.atari
@pytest.mark.parametrize('episode_limit,advance', [(200, 40), (42, 41)])
def test_real_snapshot_continuity_and_next_truncation(episode_limit, advance, monkeypatch):
    c = replace(config(), max_episode_steps=episode_limit)
    left, right = AtariRelationalEnvironment(c), AtariRelationalEnvironment(c)
    try:
        left.reset(seed=41)
        for i in range(advance):
            left.step(i % 6)
        left.backend.env.action_space.seed(51)
        left.backend.env.observation_space.seed(52)
        snapshot = pickle.loads(pickle.dumps(left.snapshot()))
        # Restore must not advance object detector memory or issue hidden steps.
        monkeypatch.setattr(right.backend.env, 'detect_objects', lambda: pytest.fail('detection on restore'))
        right.restore(snapshot)
        monkeypatch.undo()
        assert left.backend.env.action_space.sample() == right.backend.env.action_space.sample()
        np.testing.assert_array_equal(left.backend.env.observation_space.sample(),
                                      right.backend.env.observation_space.sample())
        for i in range(100):
            a, b = left.step(i % 6), right.step(i % 6)
            assert a == b
            assert [vars(o) for o in left.backend.env.objects] == [vars(o) for o in right.backend.env.objects]
            if a[-2] or a[-1]:
                assert episode_limit == 42 and i == 0 and a[-1]
                break
        # Immutable source identity guards both resume and standalone evaluation.
        snapshot['backend']['manifest']['source_hashes']['ocatari.core'] = 'changed'
        with pytest.raises(ValueError, match='hashes'):
            right.validate_snapshot(snapshot)
    finally:
        left.close()
        right.close()


@pytest.mark.ocatari
@pytest.mark.atari
def test_h5_resume_preserves_replay_rng_targets_draws_and_fit_schedule(tmp_path):
    left_env, right_env = AtariRelationalEnvironment(config()), AtariRelationalEnvironment(config())
    try:
        left = QTrainingSession(BayesianQAgent(agent_config(), left_env.encoder()), left_env)
        left.run(19)
        assert left.agent.model.model_id == 2 and not left.done
        checkpoint = left.save(tmp_path / 'agent.rrtl')
        assert left.agent.model.model_id == 2  # no extra fit at save
        right = QTrainingSession.load(checkpoint, right_env)
        left.run(13)
        right.run(13)
        assert left.history == right.history
        assert left.agent.model.model_id == right.agent.model.model_id == 4
        assert left.agent.exploration_rng.bit_generator.state == right.agent.exploration_rng.bit_generator.state
        for a, b in zip(left.agent.mcmc_rngs, right.agent.mcmc_rngs):
            assert a.bit_generator.state == b.bit_generator.state
        for a, b in zip(left.agent.replay.transitions, right.agent.replay.transitions):
            assert a.transition_id == b.transition_id
            np.testing.assert_array_equal(a.next_state.values, b.next_state.values)
        for a, b in zip(left.agent.last_fit.datasets, right.agent.last_fit.datasets):
            assert a.data_hash == b.data_hash
            np.testing.assert_array_equal(a.targets, b.targets)
        for a, b in zip(left.agent.model.posteriors, right.agent.model.posteriors):
            if a is not None:
                assert a.draws == b.draws and a.trace == b.trace
    finally:
        left_env.close()
        right_env.close()


@pytest.mark.ocatari
@pytest.mark.atari
def test_cli_train_evaluate_resume_and_corrected_rejection(tmp_path):
    root = Path(__file__).resolve().parents[1]
    path = tmp_path / 'config.json'
    path.write_text(json.dumps({'environment': replace(config(), max_episode_steps=20).resolved_dict(),
                               'agent': agent_config().to_dict(), 'interactions': 19}))
    def run(*args, success=True):
        result = subprocess.run([sys.executable, str(root / 'train_atari.py'), *map(str, args)],
                                cwd=tmp_path, capture_output=True, text=True, timeout=90)
        assert (result.returncode == 0) == success, result.stdout + result.stderr
        return result
    run('--config', path, '--variant', 'corrected', '--output', tmp_path / 'bad', success=False)
    assert not (tmp_path / 'bad').exists()
    output = tmp_path / 'trained'
    run('--config', path, '--output', output, '--eval-split', 'none')
    checkpoint = output / 'final_checkpoint.rrtl'
    before = checkpoint.read_bytes()
    agent, collector = load_checkpoint(checkpoint)
    assert collector['environment']['kind'] == 'atari_relational_v2'
    assert agent.interactions == 19 and agent.model.model_id == 2
    manifest = json.loads((output / 'manifest.json').read_text())
    runtime = manifest['runtime_environment']
    assert runtime['env_id'] == 'ALE/Pong-v5' and len(runtime['catalog']['catalog']) == 20
    assert runtime['relations']['relation_schema'] == 'pong_relational_v1'
    assert runtime['rom_sha256'] and runtime['source_hashes']
    assert manifest['environment']['reward_transform'] == 'sign'
    run('--checkpoint', checkpoint, '--eval-split', 'validation')
    assert checkpoint.read_bytes() == before and not (output / 'test.json').exists()
    assert len(json.loads((output / 'validation.json').read_text())) == 1
    resumed = tmp_path / 'resumed'
    run('--resume', checkpoint, '--output', resumed, '--steps', 13, '--eval-split', 'none')
    final, _ = load_checkpoint(resumed / 'final_checkpoint.rrtl')
    assert final.interactions == 32 and final.model.model_id == 4
    profile = json.loads((resumed / 'summary.json').read_text())['profile']['extractor']
    assert profile['backend_step_seconds'] > 0 and profile['object_conversion_seconds'] > 0
    assert 'environment_seconds' not in profile
    # The process-resumed session must equal continuation in this process.
    env = AtariRelationalEnvironment(replace(config(), max_episode_steps=20))
    try:
        expected = QTrainingSession.load(checkpoint, env)
        expected.run(13)
        assert final.exploration_rng.bit_generator.state == expected.agent.exploration_rng.bit_generator.state
        for a, b in zip(final.model.posteriors, expected.agent.model.posteriors):
            if a is not None:
                assert a.draws == b.draws and a.trace == b.trace
    finally:
        env.close()
