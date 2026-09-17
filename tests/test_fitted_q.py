"""H5 contracts: fixed datasets, terminal semantics, publication and exact resume."""

from dataclasses import replace
import json
from pathlib import Path
import pickle
import subprocess
import sys

import numpy as np
import pytest

from bayesian_rrtl.agent import BayesianQAgent, BayesianQModel
from bayesian_rrtl.checkpoint import load_checkpoint, save_checkpoint
from bayesian_rrtl.ensemble import BARTRegressor, EnsembleDraw
from bayesian_rrtl.q_config import BayesianQConfig
from bayesian_rrtl.targets import frozen_q_targets
from bayesian_rrtl.toy_mdp import TwoStepMDP
from bayesian_rrtl.training import QTrainingSession
from bayesian_rrtl.tree import Tree


def small_config(**kwargs):
    return BayesianQConfig(**dict(dict(seed=15, m=2, burn_in=5, draws=8, fit_interval=8,
                                       batch_size_per_action=16), **kwargs))


def make_agent(**kwargs):
    return BayesianQAgent(small_config(**kwargs), TwoStepMDP.encoder())


def observe(agent, *, action=0, reward=0.0, terminal=False, truncated=False, stage=0, next_state=None):
    if next_state is None and not terminal:
        next_state = TwoStepMDP.state(1)
    return agent.observe(TwoStepMDP.state(stage), action, reward, next_state, terminal, truncated,
                         raw_reward=reward * 2, episode_id=agent.interactions // 2,
                         step_id=agent.interactions % 2)


def constant_posterior(agent, value):
    batch = agent.encoder.transform([TwoStepMDP.state(0)])
    posterior = BARTRegressor(m=agent.config.m, scale=agent.config.scale).fit(
        batch, [value], np.random.default_rng(8), burn_in=0, draws=1)
    tree = Tree(float(agent.config.scale.normalize(value)) / agent.config.m)
    return replace(posterior, draws=(EnsembleDraw((tree,) * agent.config.m, 0.01),))


def test_replay_fifo_unique_ids_and_uniform_sampling():
    agent = make_agent(replay_capacity=5)
    for i in range(9):
        observe(agent, action=i % 2, reward=i, terminal=True)
    replay = agent.replay
    assert [t.transition_id for t in replay.transitions] == list(range(4, 9))
    assert replay.next_id == agent.interactions == 9
    rng = np.random.default_rng(19)
    counts = {i: 0 for i in (4, 6, 8)}
    for _ in range(1500):
        sampled = replay.sample(0, 2, rng)
        assert len({t.transition_id for t in sampled}) == 2
        for t in sampled:
            counts[t.transition_id] += 1
    assert max(abs(n - 1000) for n in counts.values()) < 80
    assert len(replay.sample(1, 99, rng)) == 2
    assert replay.sample(2, 10, rng) == ()
    assert replay.transitions[0].raw_reward == 8


@pytest.mark.parametrize('terminated,truncated,bootstrap,expected,calls', [
    (True, False, True, 2, 0), (True, True, True, 2, 0),
    (False, True, False, 2, 0), (False, True, True, 7, 1),
    (False, False, True, 7, 1)])
def test_frozen_targets_never_touch_terminal_observations(terminated, truncated, bootstrap, expected, calls):
    agent = make_agent(gamma=0.5, bootstrap_on_truncation=bootstrap)
    state = object() if terminated or (truncated and not bootstrap) else TwoStepMDP.state(1)
    transition = observe(agent, reward=2, terminal=terminated, truncated=truncated, next_state=state)
    class Target:
        count = 0
        def predict(self, batch):
            self.count += 1
            assert batch.values.tolist() == [[1]]
            return np.array([[3, 10]])
    target = Target()
    y = frozen_q_targets([transition], target, agent.encoder, gamma=0.5,
                         bootstrap_on_truncation=bootstrap)
    assert y.tolist() == [expected]
    assert not y.flags.writeable
    assert target.count == calls


def test_missing_bootstrap_observation_rejected_without_counting_transition():
    agent = make_agent()
    for truncated in (False, True):
        with pytest.raises(ValueError, match='observation'):
            agent.observe(TwoStepMDP.state(0), 0, 0, None, False, truncated,
                          raw_reward=0, episode_id=0, step_id=0)
    assert agent.interactions == 0 and len(agent.replay) == 0


def test_every_action_uses_same_frozen_target_and_atomic_publication(monkeypatch):
    agent = make_agent(gamma=0.5)
    old = BayesianQModel(agent.config.actions, (constant_posterior(agent, 1), constant_posterior(agent, 10)),
                         agent.encoder.catalog_hash, 3)
    agent.model = old
    for action in (0, 1):
        observe(agent, action=action, reward=2)
    seen = []
    def fit(regressor, batch, y, rng, **kwargs):
        assert agent.model is old
        assert not y.flags.writeable and not batch.values.flags.writeable
        seen.append(y.copy())
        return new_posterior
    new_posterior = constant_posterior(agent, 100)
    monkeypatch.setattr(BARTRegressor, 'fit', fit)
    record = agent.fit_if_due(force=True)
    np.testing.assert_allclose(seen, [[7], [7]])
    assert agent.target_model is old
    assert record.target_model_id == 3 and record.fit_id == 4
    assert record.updated_actions == (0, 1) and record.skipped_actions == ()
    assert agent.model.posteriors == (new_posterior, new_posterior)
    assert [d.transition_ids for d in record.datasets] == [(0,), (1,)]
    assert record.config_hash == agent.config.config_hash
    assert agent.fit_if_due(force=True) is None


def test_failed_fit_keeps_models_calibration_counters_and_rngs(monkeypatch):
    agent = make_agent()
    observe(agent, action=0)
    observe(agent, action=1)
    before = pickle.dumps(agent)
    original = BARTRegressor.fit
    calls = []
    def fail_second(regressor, *args, **kwargs):
        calls.append(True)
        if len(calls) == 2:
            raise RuntimeError('interrupted fit')
        return original(regressor, *args, **kwargs)
    monkeypatch.setattr(BARTRegressor, 'fit', fail_second)
    with pytest.raises(RuntimeError, match='interrupted fit'):
        agent.fit_if_due(force=True)
    assert pickle.dumps(agent) == before
    monkeypatch.setattr(BARTRegressor, 'fit', original)
    assert agent.fit_if_due(force=True).updated_actions == (0, 1)


def test_unobserved_action_stays_unfitted_or_retains_previous_model():
    agent = make_agent(replay_capacity=1)
    observe(agent, action=0, reward=1, terminal=True)
    first = agent.fit_if_due(force=True)
    posterior = agent.model.posteriors[0]
    assert first.skipped_actions == (1,)
    assert agent.model.posteriors[1] is None
    with pytest.raises(ValueError, match='no fitted posterior'):
        agent.model.predict_latent_draws(agent.encoder.transform([[]]), 1)
    observe(agent, action=1, reward=2, terminal=True)
    second = agent.fit_if_due(force=True)
    assert second.skipped_actions == (0,)
    assert agent.model.posteriors[0] is posterior


def test_scale_and_noise_calibration_frozen_across_fits_and_no_clipping():
    agent = make_agent(gamma=0.5, reward_min=-0.9, reward_max=1.1)
    assert agent.config.scale.c == pytest.approx(0.2)
    assert agent.config.scale.width == 4
    for _ in range(8):
        observe(agent, reward=0, terminal=True)
    agent.fit_if_due()
    calibration = agent.calibrations[0]
    assert calibration.residual_variance == agent.config.variance_floor
    for _ in range(8):
        observe(agent, reward=100, terminal=True)
    record = agent.fit_if_due()
    assert agent.calibrations[0] is calibration
    assert max(record.datasets[0].targets) == 100
    assert agent.model.posteriors[0].targets_outside_scale == 8
    assert agent.model.posteriors[0].scale == agent.config.scale


def compare_agents(a, b):
    assert a.interactions == b.interactions and a.epsilon == b.epsilon
    assert a.last_fit_step == b.last_fit_step and a.model.model_id == b.model.model_id
    assert a.calibrations == b.calibrations and a.training_seeds == b.training_seeds
    assert list(a.action_buffer) == list(b.action_buffer)
    for name in ('environment_rng', 'exploration_rng', 'replay_rng'):
        assert getattr(a, name).bit_generator.state == getattr(b, name).bit_generator.state
    assert [r.bit_generator.state for r in a.mcmc_rngs] == [r.bit_generator.state for r in b.mcmc_rngs]
    for model_a, model_b in ((a.model, b.model), (a.target_model, b.target_model)):
        for p, q in zip(model_a.posteriors, model_b.posteriors):
            if p is None:
                assert q is None
            else:
                assert p.draws == q.draws and p.trace == q.trace and p.data_hash == q.data_hash
    if a.last_fit:
        for x, y in zip(a.last_fit.datasets, b.last_fit.datasets):
            assert x.data_hash == y.data_hash and x.transition_ids == y.transition_ids
            np.testing.assert_array_equal(x.targets, y.targets)
    assert a.replay.next_id == b.replay.next_id and len(a.replay) == len(b.replay)
    for x, y in zip(a.replay.transitions, b.replay.transitions):
        for name in ('transition_id', 'action', 'reward', 'raw_reward', 'terminated',
                     'truncated', 'episode_id', 'step_id'):
            assert getattr(x, name) == getattr(y, name)
        for state_x, state_y in ((x.state, y.state), (x.next_state, y.next_state)):
            if state_x is None:
                assert state_y is None
            else:
                assert state_x.catalog_hash == state_y.catalog_hash
                np.testing.assert_array_equal(state_x.values, state_y.values)
                np.testing.assert_array_equal(state_x.observed, state_y.observed)


def test_mid_episode_resume_matches_next_targets_draws_policy_and_rng(tmp_path):
    config = small_config(resample_repeated_actions=True, action_buffer_capacity=2)
    uninterrupted = QTrainingSession(BayesianQAgent(config, TwoStepMDP.encoder()), TwoStepMDP(reward_noise=0.02))
    uninterrupted.run(11)
    while uninterrupted.done:
        uninterrupted.step()
    assert not uninterrupted.done
    checkpoint = uninterrupted.save(tmp_path / 'checkpoint.rrtl')
    resumed = QTrainingSession.load(checkpoint, TwoStepMDP(reward_noise=0.02), expected_config=config)
    compare_agents(uninterrupted.agent, resumed.agent)
    assert resumed.environment.snapshot() == uninterrupted.environment.snapshot()
    uninterrupted.run(25)
    resumed.run(25)
    assert uninterrupted.history == resumed.history
    assert uninterrupted.environment.snapshot() == resumed.environment.snapshot()
    compare_agents(uninterrupted.agent, resumed.agent)
    assert not resumed.agent.last_fit.datasets[0].targets.flags.writeable


def test_evaluation_does_not_mutate_agent_and_returns_raw_rewards():
    agent = make_agent()
    session = QTrainingSession(agent, TwoStepMDP())
    session.run(16)
    before = pickle.dumps(agent)
    class RawRewardMDP(TwoStepMDP):
        def step(self, action):
            state, reward, _, terminal, truncated = super().step(action)
            return state, reward, reward + 10, terminal, truncated
    rows = agent.evaluate(RawRewardMDP, agent.config.validation_seeds)
    assert rows == agent.evaluate(RawRewardMDP, agent.config.validation_seeds)
    assert all(row['return'] > 9 for row in rows)
    assert pickle.dumps(agent) == before
    with pytest.raises(ValueError, match='overlap'):
        agent.evaluate(TwoStepMDP, agent.training_seeds[:1])


def test_checkpoint_checksum_config_and_collector_validation(tmp_path):
    agent = make_agent()
    checkpoint = save_checkpoint(agent, tmp_path / 'checkpoint.rrtl')
    with pytest.raises(ValueError, match='config mismatch'):
        load_checkpoint(checkpoint, expected_config=replace(agent.config, gamma=0.7))
    with pytest.raises(ValueError, match='no collector'):
        QTrainingSession.load(checkpoint, TwoStepMDP())
    checkpoint.write_bytes(checkpoint.read_bytes()[:-1] + b'x')
    with pytest.raises(ValueError, match='checksum'):
        load_checkpoint(checkpoint)


def test_known_q_mdp_learns_optimal_actions():
    config = small_config(seed=11, m=3, burn_in=50, draws=60, fit_interval=100,
                          batch_size_per_action=128)
    agent = BayesianQAgent(config, TwoStepMDP.encoder())
    QTrainingSession(agent, TwoStepMDP()).run(600)
    batch = agent.encoder.transform([TwoStepMDP.state(0), TwoStepMDP.state(1)])
    predicted = agent.model.predict(batch)
    np.testing.assert_allclose(predicted, TwoStepMDP.optimal_q(config.gamma), atol=0.12, rtol=0)
    assert predicted.argmax(axis=1).tolist() == [0, 0]
    assert all(p is not None for p in agent.model.posteriors)


def test_cli_train_resume_and_evaluate_existing_checkpoint(tmp_path):
    root = Path(__file__).resolve().parents[1]
    config = small_config(m=1, burn_in=2, draws=3)
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(config.to_dict()))
    def run(*args):
        result = subprocess.run([sys.executable, str(root / 'train_bayesian.py'), *map(str, args)],
                                cwd=tmp_path, capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, result.stdout + result.stderr
    first, resumed = tmp_path / 'first', tmp_path / 'resumed'
    run('--config', path, '--output', first, '--steps', 11)
    original = (first / 'final_checkpoint.rrtl').read_bytes()
    assert not (first / 'test.json').exists()
    run('--resume', first / 'final_checkpoint.rrtl', '--output', resumed, '--steps', 10)
    assert json.loads((resumed / 'summary.json').read_text())['interactions'] == 21
    run('--checkpoint', first / 'final_checkpoint.rrtl', '--eval-split', 'test')
    assert (first / 'final_checkpoint.rrtl').read_bytes() == original
    assert [r['seed'] for r in json.loads((first / 'test.json').read_text())] == list(config.test_seeds)


@pytest.mark.parametrize('kwargs', [{'draws': 0}, {'fit_interval': 0}, {'gamma': 1},
                                   {'reward_min': 0, 'reward_max': 0}, {'actions': ('x', 'x')},
                                   {'test_seeds': (100001,)}, {'variance_floor': 0}])
def test_invalid_q_configuration(kwargs):
    with pytest.raises(ValueError):
        BayesianQConfig(**kwargs)
