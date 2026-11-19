import copy
import math
from pathlib import Path
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from bayesian_rrtl.agent import BayesianQModel
from bayesian_rrtl.encoding import RelationalEncoder
from bayesian_rrtl.ensemble import BARTRegressor, BoundedStructureKernel, StructureLimitError
from bayesian_rrtl.exact import enumerate_trees, exact_posterior
from bayesian_rrtl.online.learners import LearningConfig, OnlineBatchAgent, online_bart_config
from bayesian_rrtl.online.session import OnlineSession
from bayesian_rrtl.priors import TreePrior
from bayesian_rrtl.proposals import GrowPruneKernel, log_acceptance_ratio
from bayesian_rrtl.tree import Tree
from experiments.breakout_bart_stability import VARIANTS, protocols
from test_bayesian_kernel import tiny_data
from test_pong_online import TinyEnvironment, campaign


def test_structure_distance_counts_all_removed_splits_and_ignores_leaf_values():
    batch, _, rules = tiny_data()
    trees = enumerate_trees(batch, rules, TreePrior(max_depth=2))
    deep = next(t for t in trees if t.structure_distance(Tree()) == 2)
    assert Tree().structure_distance(deep) == deep.structure_distance(Tree()) == 2
    assert deep.structure_distance(copy.deepcopy(deep)) == 0
    assert Tree(1).structure_distance(Tree(-20)) == 0
    assert all(a.structure_distance(b) == b.structure_distance(a) for a in trees for b in trees)


@pytest.mark.parametrize('limit', [0, 1, 2])
def test_every_retained_tree_respects_anchor_distance_with_fresh_leaves(limit):
    batch, y, rules = tiny_data()
    prior = TreePrior(max_depth=2)
    anchor = next(t for t in enumerate_trees(batch, rules, prior) if not t.is_leaf)
    posterior = BARTRegressor(m=2, tree_prior=prior).fit(batch, y, np.random.default_rng(715),
        initial_trees=(anchor, Tree()), max_structure_distance=limit,
        burn_in=10, draws=30, verify_cache=True)
    for draw in posterior.draws:
        for old, new in zip((anchor, Tree()), draw.trees):
            assert new.structure_distance(old) <= limit
            new.validate_support(batch, rules, max_depth=2)
    assert posterior.max_cache_error < 1e-12
    assert max(posterior.structure_distances) <= limit
    assert posterior.diagnostics()['reset_trees'] == 0
    if limit == 0:
        assert posterior.distance_rejections > 0
        assert posterior.draws[0].trees[0] != posterior.draws[-1].trees[0]


def test_support_repair_uses_same_budget_and_fails_before_consuming_rng():
    batch, y, rules = tiny_data()
    prior = TreePrior(max_depth=2)
    anchor = next(t for t in enumerate_trees(batch, rules, prior) if t.structure_distance(Tree()) == 2)
    constant = batch.take([0, 0, 0])
    regressor = BARTRegressor(m=1, tree_prior=prior)
    rng = np.random.default_rng(3)
    before = copy.deepcopy(rng.bit_generator.state)
    with pytest.raises(StructureLimitError) as exc:
        regressor.fit(constant, y, rng, initial_trees=(anchor,), max_structure_distance=1,
                      burn_in=0, draws=2)
    assert exc.value.required_distance == 2 and rng.bit_generator.state == before
    posterior = regressor.fit(constant, y, rng, initial_trees=(anchor,), max_structure_distance=2,
                              burn_in=0, draws=2)
    assert posterior.support_repairs == (2,)
    assert all(d.trees[0].is_leaf for d in posterior.draws)


def test_bounded_kernel_preserves_posterior_restricted_to_structural_neighbourhood():
    batch, y, rules = tiny_data()
    prior = TreePrior(alpha_tree=.6, max_depth=2)
    all_trees = enumerate_trees(batch, rules, prior)
    anchor = next(t for t in all_trees if not t.is_leaf)
    trees = [t for t in all_trees if t.structure_distance(anchor) <= 1]
    index = {t.structure_key(): i for i, t in enumerate(trees)}
    probability = exact_posterior(trees, batch, y, .1, .0625, prior, rules)
    matrix = np.zeros((len(trees), len(trees)))
    base = GrowPruneKernel(batch, rules, prior)
    for i, tree in enumerate(trees):
        for proposal in base.all_proposals(tree):
            kernel = BoundedStructureKernel(SimpleNamespace(rules=rules, propose=lambda *_: proposal), anchor, 1)
            bounded = kernel.propose(tree, None)
            mass = math.exp(proposal.log_q_forward)
            ratio = log_acceptance_ratio(tree, bounded, batch, y, .1, .0625, prior, rules)
            accepted = 0 if bounded.impossible else min(1., math.exp(min(0., ratio)))
            j = i if bounded.impossible else index[bounded.candidate.structure_key()]
            matrix[i, j] += mass * accepted
            matrix[i, i] += mass * (1 - accepted)
    np.testing.assert_allclose(matrix.sum(1), 1., atol=1e-14)
    flow = probability[:, None] * matrix
    np.testing.assert_allclose(flow, flow.T, atol=1e-14)
    np.testing.assert_allclose(probability @ matrix, probability, atol=1e-14)


def small_agent(**estimator):
    config = LearningConfig(('A', 'B'), seed=715, gamma=.5, warmup_interactions=0,
                            replay_capacity=16, fit_interval=1, batch_size_per_action=4)
    encoder = TinyEnvironment().encoder()
    return OnlineBatchAgent(config, encoder, learner='bart',
        estimator={'m': 1, 'burn_in': 1, 'draws': 2, **estimator})


def observe(agent, *, terminal=False, state=None):
    state = [] if state is None else state
    agent.observe(state, 0, 1., state, terminal, False, raw_reward=1., episode_id=0, step_id=0)


def test_four_fit_target_schedule_targets_and_monotone_model_ids():
    agent = small_agent(target_update_interval=4)
    models = {0: agent.model}
    for fit_id in range(1, 10):
        observe(agent)
        expected_id = ((fit_id - 1) // 4) * 4
        expected = 1 + .5 * models[expected_id].predict(agent.encoder.transform([[]])).max()
        record = agent.fit_if_due()
        assert (record.fit_id, record.target_model_id) == (fit_id, expected_id)
        assert agent.target_model is models[expected_id]
        np.testing.assert_allclose(record.datasets[0].targets, expected)
        models[fit_id] = agent.model


def test_support_limit_retains_previous_action_posterior():
    batch, y, rules = tiny_data()
    prior = TreePrior(max_depth=2)
    anchor = next(t for t in enumerate_trees(batch, rules, prior) if not t.is_leaf)
    agent = small_agent(warm_start=True, max_structure_distance=0)
    agent.encoder = RelationalEncoder(batch.catalog)
    old = BARTRegressor(m=1, tree_prior=prior, scale=agent.config.scale).fit(
        batch, y, np.random.default_rng(1), initial_trees=(anchor,), max_structure_distance=0,
        burn_in=0, draws=2)
    agent.model = BayesianQModel(agent.config.actions, (old, None), batch.catalog_hash)
    observe(agent, state=batch.take([0]))
    record = agent.fit_if_due()
    assert agent.model.posteriors[0] is old
    assert record.skipped_actions == (0, 1) and record.updated_actions == ()
    assert record.diagnostics['A']['reason'] == 'support_repair_exceeds_limit'


def test_failed_delayed_target_fit_does_not_publish_or_advance_rngs(monkeypatch):
    agent = small_agent(target_update_interval=4)
    observe(agent)
    agent.fit_if_due()
    observe(agent)
    before_model, before_target = agent.model, agent.target_model
    before_rngs = pickle.dumps((agent.replay_rng, agent.mcmc_rngs))
    def fail(*args):
        raise RuntimeError('test interruption')
    monkeypatch.setattr(agent, 'fit_action', fail)
    with pytest.raises(RuntimeError, match='interruption'):
        agent.fit_if_due()
    assert agent.model is before_model and agent.target_model is before_target
    assert agent.fit_count == 1 and agent.last_fit_step == 1
    assert pickle.dumps((agent.replay_rng, agent.mcmc_rngs)) == before_rngs


@pytest.mark.parametrize('estimator', [VARIANTS['bounded_structure'], VARIANTS['slow_target']])
def test_online_checkpoint_resume_preserves_next_fits_and_predictions(tmp_path, estimator):
    cell = campaign(tmp_path / 'base', 'bart')
    cell['estimator'].update(estimator)
    uninterrupted = OnlineSession(cell, TinyEnvironment())
    for _ in range(23):
        uninterrupted.step()
    payload = pickle.loads(pickle.dumps(uninterrupted.state_dict()))
    resumed = OnlineSession.restore(payload, cell, TinyEnvironment())
    for _ in range(40):
        first, second = uninterrupted.step(), resumed.step()
        # Durations are machine observations; compare all scientific diagnostics.
        for row in (first, second):
            if row['fit']:
                for details in row['fit']['actions'].values():
                    details.pop('elapsed_seconds', None)
        assert first == second
        assert uninterrupted.agent.model.model_id == resumed.agent.model.model_id
        assert uninterrupted.agent.target_model.model_id == resumed.agent.target_model.model_id
        for a, b in zip(uninterrupted.agent.model.posteriors, resumed.agent.model.posteriors):
            if a is not None:
                assert a.draws == b.draws


@pytest.mark.parametrize('options', [
    {'max_structure_distance': 2}, {'warm_start': True, 'max_structure_distance': -1},
    {'warm_start': True, 'max_structure_distance': True}, {'target_update_interval': 0},
    {'target_update_interval': True},
])
def test_invalid_stability_options(options):
    with pytest.raises(ValueError):
        online_bart_config(LearningConfig(('A',), seed=1), options)


def test_campaign_has_only_ten_separate_trials_and_same_reference_except_estimator():
    import json
    from experiments.pong_analysis_runtime import checked_read
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / 'configs/breakout_bart_stability_1m_20261114.json').read_text())
    historical = checked_read(root / 'results_cluster/rrl_breakout_bart_cadence_500k_20261112_01'
                              / 'pilot/interval_1024/campaign.json')
    variants = protocols(config, historical)
    a, b = [copy.deepcopy(v) for v in variants.values()]
    assert len(a['cells']) + len(b['cells']) == 10
    assert a['games']['Breakout']['max_decisions'] == 1_000_000
    for p in (a, b):
        p.pop('name')
        p.pop('learners')
    assert a == b
    assert variants['bounded_structure']['learners']['bart'].get('target_update_interval', 1) == 1
    assert not variants['slow_target']['learners']['bart']['warm_start']
    config['training_seeds'].append(733106)
    with pytest.raises(ValueError, match='ten trainings'):
        protocols(config, historical)
