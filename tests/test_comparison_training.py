import copy
from dataclasses import replace
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from bayesian_rrtl.agent import BayesianQAgent
from bayesian_rrtl.atari import AtariConfig
from bayesian_rrtl.baseline_adapter import BaselineLearnerConfig, RelationalBaselineAgent
from bayesian_rrtl.baseline_training import BaselineTrainingSession
from bayesian_rrtl.comparison import (BayesianComparisonSession, ComparisonPolicyConfig,
                                      PongEnvironmentFactory, make_comparison_session)
from bayesian_rrtl.encoding import RelationKey, RelationalEncoder
from bayesian_rrtl.q_config import BayesianQConfig
from bayesian_rrtl.targets import frozen_q_targets
from bayesian_rrtl.training import QTrainingSession
from preprocessing import Fact


class TraceEnvironment:
    actions = ('left', 'right')

    def __init__(self, *, terminal=True, length=3):
        self.terminal, self.length = terminal, length
        self.resets, self.steps, self.closed = [], 0, False
        self.episode_step = 0

    def encoder(self):
        return RelationalEncoder([RelationKey('comparative', 'x', 'a', 'b')])

    def reset(self, *, seed):
        self.resets.append(seed)
        self.episode_step = 0
        return [Fact('comparative', 'less', 'x', 'a', 'b')]

    def step(self, action):
        self.steps += 1
        self.episode_step += 1
        done = self.episode_step == self.length
        raw = 2. if self.episode_step == 1 else -1.
        return ([Fact('comparative', 'more', 'x', 'a', 'b')], float(np.sign(raw)) + .1,
                raw, bool(done and self.terminal), bool(done and not self.terminal))

    def close(self):
        self.closed = True


def sessions(*, policy=ComparisonPolicyConfig(), terminal=True, length=3, fit_interval=1000):
    left, right = [TraceEnvironment(terminal=terminal, length=length) for _ in range(2)]
    baseline = BaselineTrainingSession(RelationalBaselineAgent(left.actions, left.encoder(),
        BaselineLearnerConfig(seed=61, gamma=.5, eta_q=.25)), left, policy)
    bart_config = BayesianQConfig(actions=right.actions, seed=61, gamma=.5,
        reward_min=-.9, reward_max=1.1, epsilon_decay_steps=50000,
        validation_seeds=policy.validation_seeds, test_seeds=policy.test_seeds,
        fit_interval=fit_interval, m=1, burn_in=1, draws=2, batch_size_per_action=8)
    bart = BayesianComparisonSession(BayesianQAgent(bart_config, right.encoder()), right, policy)
    return baseline, bart


def common(row):
    return {key: value for key, value in row.items() if key != 'learner_metrics'}


def snapshot(session):
    return pickle.dumps({k: v for k, v in vars(session).items() if k != 'environment_factory'})


def test_streams_schedule_and_greedy_ties_match_h5():
    baseline, bart = sessions()
    assert baseline.environment_rng.bit_generator.state == bart.agent.environment_rng.bit_generator.state
    assert baseline.exploration_rng.bit_generator.state == bart.agent.exploration_rng.bit_generator.state
    for step in (0, 1, 12345, 49999, 50000, 100000):
        baseline.agent.interactions = bart.agent.interactions = step
        assert baseline.epsilon == bart.agent.epsilon
        rng = baseline.exploration_rng
        for _ in range(100):
            assert baseline.agent.select_action([], epsilon=baseline.epsilon, rng=rng) == (
                bart.agent.select_action([]))
        assert rng.bit_generator.state == bart.agent.exploration_rng.bit_generator.state
    assert baseline.epsilon == .1
    # At epsilon=0 evaluation must neither explore nor consume the exploration coin.
    baseline.exploration_rng = np.random.default_rng(7)
    bart.agent.exploration_rng = np.random.default_rng(7)
    choices = []
    for _ in range(100):
        action = baseline.agent.select_action([], epsilon=0., rng=baseline.exploration_rng)
        assert action == bart.agent.select_action([], explore=False)
        choices.append(action)
    assert set(choices) == {0, 1}


def test_reserved_episode_seeds_are_skipped_without_consuming_policy_rng():
    baseline, _ = sessions()
    probe = copy.deepcopy(baseline.environment_rng)
    forbidden = tuple(int(probe.integers(0, 2**32)) for _ in range(2))
    policy = ComparisonPolicyConfig(validation_seeds=forbidden[:1], test_seeds=forbidden[1:])
    baseline, bart = sessions(policy=policy)
    expected = int(probe.integers(0, 2**32))
    before = baseline.exploration_rng.bit_generator.state
    assert baseline.episode_seed() == bart.agent.episode_seed() == expected
    assert baseline.training_seeds == bart.agent.training_seeds == [expected]
    assert baseline.exploration_rng.bit_generator.state == before


@pytest.mark.parametrize('terminal', [False, True])
def test_both_collectors_fixed_trace_point_budget_flags_and_logs(monkeypatch, terminal):
    baseline, bart = sessions(terminal=terminal)
    for session in (baseline, bart):
        monkeypatch.setattr(session.agent, 'select_action', lambda *a, **kw: 1)
        session.run(1)  # Raw point is not a terminal; exactly one step/update.
        assert not session.done and session.environment.steps == session.interactions == 1
        assert session.history[0]['raw_reward'] == 2.
        assert session.history[0]['reward'] == 1.1
        assert session.history[0]['return_scope'] == 'partial'
        assert len(session.environment.resets) == 1
        assert session.metrics()['mean_completed_game_raw_return'] is None
        session.run(2)
        assert session.done and session.history[-1]['terminated'] == terminal
        assert session.history[-1]['truncated'] != terminal
        assert session.history[-1]['return_scope'] == ('complete_game' if terminal else 'horizon_limited')
        session.run(1)
        assert not session.done and len(session.environment.resets) == 2
        assert session.step_id == 1 and session.episode_id == 1
        metrics = session.metrics()
        assert metrics['mean_completed_game_raw_return'] == (0. if terminal else None)
        assert metrics['partial_episode']['raw_episode_return'] == 2.
        assert len(session.episode_rows()) == 2
        assert len(session.episode_history) == 1
        assert [r['transition_id'] for r in session.history] == list(range(4))
    assert [common(r) for r in baseline.history] == [common(r) for r in bart.history]
    assert baseline.agent.query_counts['update']['queries'] == 4
    assert len(bart.agent.replay) == 4 and bart.agent.model.model_id == 0
    transition = bart.agent.replay.transitions[2]
    assert (transition.next_state is None) == terminal
    td = baseline.history[2]['learner_metrics']['td']
    assert td['bootstrap'] == (0. if terminal else 1.)
    # Final observation supplies the target on truncation; terminal never queries it.
    calls = []
    def predict(batch):
        calls.append(batch)
        return np.full((len(batch), 2), 6.)
    targets = frozen_q_targets([transition], SimpleNamespace(predict=predict), bart.agent.encoder, gamma=.5)
    assert targets[0] == pytest.approx(-.9 if terminal else 2.1)
    assert bool(calls) != terminal


def test_bart_decorator_preserves_h5_loop_fits_and_budget_continuity():
    _, decorated = sessions(length=10, fit_interval=3)
    original = QTrainingSession(BayesianQAgent(decorated.agent.config, decorated.agent.encoder),
                                TraceEnvironment(length=10))
    decorated.run(2)
    assert decorated.agent.model.model_id == 0  # No fit forced at budget boundary.
    decorated.run(3)
    original.run(5)
    assert decorated.agent.model.model_id == 1
    for old, new in zip(original.history, decorated.history):
        assert old == {key: new[key] for key in old}
    for name in ('environment_rng', 'exploration_rng', 'replay_rng'):
        assert getattr(decorated.agent, name).bit_generator.state == getattr(original.agent, name).bit_generator.state
    for left, right in zip(decorated.agent.mcmc_rngs, original.agent.mcmc_rngs):
        assert left.bit_generator.state == right.bit_generator.state
    for left, right in zip(decorated.agent.model.posteriors, original.agent.model.posteriors):
        assert left.draws == right.draws and left.trace == right.trace
    for left, right in zip(decorated.agent.last_fit.datasets, original.agent.last_fit.datasets):
        assert left.data_hash == right.data_hash
        np.testing.assert_array_equal(left.targets, right.targets)
    for left, right in zip(decorated.agent.replay.transitions, original.agent.replay.transitions):
        assert left.transition_id == right.transition_id
        np.testing.assert_array_equal(left.state.values, right.state.values)
        np.testing.assert_array_equal(left.next_state.values, right.next_state.values)
    assert decorated.history[2]['learner_metrics']['fit_performed']
    assert decorated.history[2]['learner_metrics']['fit_diagnostics']
    assert not decorated.history[4]['learner_metrics']['fit_performed']
    assert decorated.environment.resets == original.environment.resets


@pytest.mark.parametrize('kind', [0, 1])
@pytest.mark.parametrize('terminal,length,horizon,scope,total', [
    (True, 3, 5, 'complete_game', 0.), (False, 3, 5, 'horizon_limited', 0.),
    (True, 10, 2, 'horizon_limited', 1.),
])
def test_frozen_evaluation_raw_return_private_rng_and_no_training_mutation(
        kind, terminal, length, horizon, scope, total):
    session = sessions(fit_interval=2)[kind]
    session.run(2)  # BART has a fitted posterior; baseline has nonzero Q/statistics.
    before = snapshot(session)
    envs = []
    def factory():
        env = TraceEnvironment(terminal=terminal, length=length)
        envs.append(env)
        return env
    result = session.evaluate((100001, 100002), horizon=horizon, environment_factory=factory)
    assert snapshot(session) == before
    assert result == session.evaluate((100001, 100002), horizon=horizon, environment_factory=factory)
    assert snapshot(session) == before
    assert len(envs) == 4 and all(env.closed for env in envs)
    assert all(row['return_scope'] == scope and row['return'] == total for row in result['episodes'])
    assert all(row['points_won'] == 2. and row['points_lost'] == 2. - total
               for row in result['episodes'])
    assert result['median_raw_return'] == total
    assert result['std_raw_return'] == result['iqr_raw_return'] == 0.
    assert result['mean_raw_return'] == total
    assert result['truncation_fraction'] == (1. if scope == 'horizon_limited' else 0.)
    assert result['completed_games'] == (2 if scope == 'complete_game' else 0)
    assert all(row['epsilon'] == 0. for row in result['episodes'])


@pytest.mark.parametrize('kind', [0, 1])
def test_eval_overlap_same_environment_errors_and_cleanup(kind):
    session = sessions()[kind]
    session.run(1)
    seed = session.history[0]['episode_seed']
    before = snapshot(session)
    def forbidden():
        pytest.fail('Must reject seeds before creating an environment')
    for seeds in ((seed,), (), (1, 1), (-1,)):
        with pytest.raises(ValueError):
            session.evaluate(seeds, horizon=2, environment_factory=forbidden)
    with pytest.raises(ValueError, match='reuse'):
        session.evaluate((100001,), horizon=2, environment_factory=lambda: session.environment)
    assert snapshot(session) == before and not session.environment.closed
    bad = TraceEnvironment()
    bad.actions = ('wrong',)
    with pytest.raises(ValueError, match='actions/encoder'):
        session.evaluate((100001,), horizon=2, environment_factory=lambda: bad)
    assert bad.closed and snapshot(session) == before


@pytest.mark.parametrize('kind', [0, 1])
def test_evaluation_rejects_changed_semantics_even_with_same_encoder(kind):
    session = sessions()[kind]
    session.environment.config = AtariConfig(game='Pong', extractor='legacy_visual_pong_v1')
    bad = TraceEnvironment()
    bad.config = replace(session.environment.config, repeat_action_probability=.25)
    before = snapshot(session)
    with pytest.raises(ValueError, match='semantics'):
        session.evaluate((100001,), horizon=2, environment_factory=lambda: bad)
    assert bad.closed and snapshot(session) == before


@pytest.mark.parametrize('kind', [0, 1])
def test_collector_does_not_read_terminal_next_state(monkeypatch, kind):
    session = sessions()[kind]
    def terminal(action):
        return object(), -1., -1., True, False
    monkeypatch.setattr(session.environment, 'step', terminal)
    row = session.step()
    assert row['terminated'] and session.done
    assert row['return_scope'] == 'complete_game'
    if kind == 0:
        assert row['learner_metrics']['td']['target'] == -1.
        assert session.agent.query_counts['bootstrap']['queries'] == 0
    else:
        assert session.agent.replay.transitions[-1].next_state is None


def test_factory_rejects_incompatible_pairs_before_creating_ale(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('Invalid setup must not open ALE')
    monkeypatch.setattr('bayesian_rrtl.comparison.AtariRelationalEnvironment', forbidden)
    env_config = AtariConfig(game='Pong', extractor='legacy_visual_pong_v1')
    spec = ('NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE')
    config = BayesianQConfig(actions=spec, gamma=.99, reward_min=-.9, reward_max=1.1,
                             epsilon_decay_steps=50000)
    for changes in ({'epsilon_decay_steps': 1000}, {'resample_repeated_actions': True},
                    {'gamma': .8}, {'reward_min': -1.}, {'actions': ('wrong',)}):
        with pytest.raises(ValueError):
            make_comparison_session('bart', env_config, replace(config, **changes))
    for config in (BaselineLearnerConfig(gamma=.5), BaselineLearnerConfig(seed=100001)):
        with pytest.raises(ValueError):
            make_comparison_session('corrected_common', env_config, config)
    with pytest.raises(ValueError):
        make_comparison_session('unknown', env_config, BaselineLearnerConfig())
    for config in (replace(env_config, frameskip=2), replace(env_config, include_incomplete_states=True)):
        with pytest.raises(ValueError):
            PongEnvironmentFactory(config)


@pytest.mark.parametrize('kind', [0, 1])
def test_zero_budget_invalid_budget_and_detached_metrics(kind):
    session = sessions()[kind]
    before = snapshot(session)
    assert session.run(0) == []
    for budget in (-1, True, 1.5):
        with pytest.raises(ValueError):
            session.run(budget)
    assert snapshot(session) == before
    session.run(1)
    rows = session.episode_rows()
    rows[0]['raw_episode_return'] = 1000.
    assert session.history[0]['raw_episode_return'] == 2.
