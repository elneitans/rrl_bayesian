"""Gate A: historical TD/F-test equivalence, including actual tree growth."""

import copy
from dataclasses import asdict, replace
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from bayesian_rrtl.baseline import CorrectedRRLAgent
from bayesian_rrtl.baseline_adapter import (
    BaselineLearnerConfig, RelationalBaselineAgent, normalize_facts,
)
from bayesian_rrtl.config import BaselineConfig
from bayesian_rrtl.encoding import RelationKey, RelationalEncoder
from bayesian_rrtl.games.pong import PONG_ACTIONS, PongRelationsV1
from bayesian_rrtl.objects import MultiGameRelations
from preprocessing import Fact


def tree_state(node):
    """Compare every stored field, excluding classes and cyclic parent pointers."""
    branches = {'more_branch', 'same_branch', 'less_branch', 'true_branch', 'false_branch'}
    state = {k: copy.deepcopy(v) for k, v in vars(node).items()
             if k not in branches | {'children', 'parent'}}
    for child in node.children:
        assert child.parent is node
    state['branches'] = {k: None if getattr(node, k) is None else tree_state(getattr(node, k))
                         for k in branches if hasattr(node, k)}
    state['children'] = [tree_state(child) for child in node.children]
    return state


def signal(i):
    x = ('less', 'more')[i % 2]
    y = ('less', 'more')[(i // 2) % 2]
    state = [Fact('comparative', x, 'x', 'player_t', 'ball_t'),
             Fact('comparative', y, 'y', 'player_t', 'ball_t')]
    return state, (-4 if x == 'less' else 4) + (-1 if y == 'less' else 1)


def pong_agent(**kwargs):
    return RelationalBaselineAgent(PONG_ACTIONS, PongRelationsV1().encoder(),
                                   BaselineLearnerConfig(**kwargs))


@pytest.mark.parametrize('criterion', ['p-value', 'f-ratio', 'max-q-value'])
@pytest.mark.parametrize('inherit', [False, True])
def test_historical_gate_with_real_splits(tmp_path, criterion, inherit):
    config = BaselineLearnerConfig(eta_q=1., gamma=0., min_sample_size=6, max_depth=3,
                                  significance_level=.05, best_literal_criteria=criterion,
                                  inherit_q_values=inherit)
    historical = CorrectedRRLAgent(BaselineConfig(game='Pong', **asdict(config)), tmp_path)
    catalog = tuple(RelationKey(lit.type, lit.name, lit.obj1, lit.obj2, lit.obj3)
                    for lit in historical.refinements)
    adapted = RelationalBaselineAgent(historical.valid_actions, RelationalEncoder(catalog),
                                      config, ordered_catalog=catalog)
    assert adapted.literals == tuple(historical.refinements)
    inherited = 0
    for i in range(120):
        state, reward = signal(i)
        leaf = adapted.root_nodes['NOOP'].get_state_leaf(state)
        before = adapted.n_splits
        actual = adapted.observe(state, 0, reward, None, True, False)
        expected = historical.update_transition(state, 0, reward, [], True, False)
        historical.i_step += 1  # H0 counts decisions in its collector.
        assert actual == expected
        assert adapted.interactions == historical.i_step == i + 1
        assert adapted.n_splits == historical.n_splits
        if adapted.n_splits > before:
            assert leaf.children
            assert all(child.q_value == (actual.new_q if inherit else 0.) for child in leaf.children)
            inherited += 1
        np.testing.assert_array_equal(adapted.predict_q(state),
                                      [root.predict(state) for root in historical.root_nodes.values()])
        for action in adapted.actions:
            assert tree_state(adapted.root_nodes[action]) == tree_state(historical.root_nodes[action])
    assert adapted.n_splits == inherited == 3


def test_nonterminal_and_truncation_trace_matches_h0(tmp_path):
    config = BaselineLearnerConfig(eta_q=.25, gamma=.5, min_sample_size=6)
    reference = CorrectedRRLAgent(BaselineConfig(game='Pong', **asdict(config)), tmp_path)
    catalog = tuple(RelationKey(lit.type, lit.name, lit.obj1, lit.obj2, lit.obj3)
                    for lit in reference.refinements)
    agent = RelationalBaselineAgent(reference.valid_actions, RelationalEncoder(catalog), config,
                                    ordered_catalog=catalog)
    for i in range(48):
        state, reward = signal(i)
        following, _ = signal(i + 1)
        action, terminal, truncated = i % 6, i % 7 == 0, i % 5 == 0
        assert agent.observe(state, action, reward, following, terminal, truncated) == (
            reference.update_transition(state, action, reward, following, terminal, truncated))
        for name in agent.actions:
            assert tree_state(agent.root_nodes[name]) == tree_state(reference.root_nodes[name])


@pytest.mark.parametrize('value', [True, False, np.bool_(True), np.bool_(False), 'True', 'False'])
def test_boolean_boundary_is_semantic_and_nonmutating(value):
    facts = frozenset([Fact('logical', value, 'present', 'ball_t'),
                       Fact('comparative', 'more', 'y', 'player_t', 'player_t-1')])
    before = copy.deepcopy(facts)
    normalized = normalize_facts(facts)
    assert facts == before
    assert [f.value for f in normalized if f.type == 'logical'] == [str(value)]
    encoder = PongRelationsV1().encoder()
    original, canonical = encoder.transform([facts]), encoder.transform([normalized])
    np.testing.assert_array_equal(original.values, canonical.values)
    np.testing.assert_array_equal(original.observed, canonical.observed)
    absent = encoder.catalog.index(RelationKey('logical', 'present', 'enemy_t'))
    assert not canonical.observed[0, absent]
    assert canonical.values[0, absent] == -1


def test_environment_factory_and_fixed_order_without_emulator(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('Adapter must not construct H0, preprocess images or create an emulator')
    monkeypatch.setattr(CorrectedRRLAgent, '__init__', forbidden)
    monkeypatch.setattr('gymnasium.make', forbidden)
    monkeypatch.setattr('preprocessing.PongPreprocessor', forbidden)
    actions = tuple(reversed(PONG_ACTIONS))
    encoder = PongRelationsV1().encoder()
    env = SimpleNamespace(actions=actions, spec=SimpleNamespace(actions=actions), encoder=lambda: encoder)
    agent = RelationalBaselineAgent.from_environment(env)
    assert agent.actions == actions
    assert agent.catalog == encoder.catalog
    assert len(agent.literals) == 20
    assert RelationKey('comparative', 'y', 'player_t', 'player_t-1') in agent.catalog
    assert sum(key.name == 'present' for key in agent.catalog) == 6
    for i, root in enumerate(agent.root_nodes.values()):
        root.q_value = i
    np.testing.assert_array_equal(agent.predict_q([]), np.arange(6))
    assert agent.select_action([]) == 5
    contract = agent.contract()
    agent.observe([], 0, 0., None, True, False)
    assert agent.contract() == contract
    env.actions = PONG_ACTIONS
    with pytest.raises(ValueError, match='actions'):
        RelationalBaselineAgent.from_environment(env)


def grow_presence_tree(agent):
    for i in range(24):
        value = bool(i % 2)
        state = [Fact('logical', value, 'present', 'ball_t'),
                 Fact('comparative', 'same', 'x', 'player_t', 'enemy_t')]
        agent.observe(state, 0, 1. if value else -1., None, True, False)
    root = agent.root_nodes['NOOP']
    assert root.literal.name == 'present'
    assert agent.n_splits == 1
    return root


def test_true_false_missing_route_and_fallback_counts():
    agent = pong_agent(eta_q=1., gamma=.5, min_sample_size=6, significance_level=.05)
    root = grow_presence_tree(agent)
    true = [Fact('logical', True, 'present', 'ball_t')]
    false = [Fact('logical', False, 'present', 'ball_t')]
    assert root.true_branch.q_value == 1.
    assert root.false_branch.q_value == -1.
    root.q_value = 7.
    assert agent.predict_q(true)[0] == 1.
    assert agent.predict_q(false)[0] == -1.
    assert agent.predict_q([])[0] == 7.
    assert agent.predict_q([Fact('comparative', 'same', 'x', 'player_t', 'enemy_t')])[0] == 7.
    assert agent.query_counts['prediction'] == {'queries': 24, 'internal_fallbacks': 2}
    before = copy.deepcopy(root.refinements_stats)
    update = agent.observe([], 0, .1, [], False, True)
    assert update.target == 3.6
    assert root.q_value == update.new_q
    assert root.refinements_stats == before
    assert agent.query_counts['update'] == {'queries': 25, 'internal_fallbacks': 1}
    assert agent.query_counts['bootstrap'] == {'queries': 6, 'internal_fallbacks': 1}
    assert root.false_branch.q_value == -1.


def test_presence_only_keeps_historical_no_split_guard_and_empty_td():
    agent = pong_agent(eta_q=1., gamma=0., min_sample_size=1, significance_level=.5)
    literal = next(lit for lit in agent.literals if lit.name == 'present' and lit.obj1 == 'ball_t')
    for i in range(20):
        agent.observe([Fact('logical', bool(i % 2), 'present', 'ball_t')],
                      0, float(i % 2), None, True, False)
    root = agent.root_nodes['NOOP']
    assert root.refinements_stats[literal]['Total']['n'] == 20
    assert root.refinements_stats[literal]['Total']['J'] > 0
    assert agent.n_splits == 0
    before = copy.deepcopy(root.refinements_stats)
    assert agent.observe([], 0, .1, None, True, False).new_q == pytest.approx(.1)
    assert root.refinements_stats == before
    assert agent.interactions == 21


@pytest.mark.parametrize('criterion', ['p-value', 'f-ratio', 'max-q-value'])
def test_constant_zero_variance_then_recover(criterion):
    agent = pong_agent(eta_q=1., gamma=0., min_sample_size=1, significance_level=.05,
                       best_literal_criteria=criterion)
    for i in range(8):
        state, _ = signal(i)
        agent.observe(state, 0, 0., None, True, False)
    assert agent.n_splits == 0
    for i in range(100):
        state, reward = signal(i)
        agent.observe(state, 0, reward, None, True, False)
    assert agent.n_splits > 0


@pytest.mark.parametrize('terminal,truncated,bootstrap,expected', [
    (True, False, True, .1), (True, True, True, .1),
    (False, True, False, .1), (False, True, True, 3.1), (False, False, True, 3.1),
])
def test_flags_reward_already_transformed_and_one_update(terminal, truncated, bootstrap, expected):
    agent = pong_agent(eta_q=.25, gamma=.5, bootstrap_on_truncation=bootstrap)
    agent.root_nodes['NOOP'].q_value = 4.
    agent.root_nodes['FIRE'].q_value = 6.
    will_bootstrap = not terminal and (not truncated or bootstrap)
    # An opaque sentinel must never be read when there is no bootstrap.
    update = agent.observe([], 0, .1, [] if will_bootstrap else object(), terminal, truncated)
    assert update.target == expected
    assert update.new_q == 4 + .25 * (expected - 4)
    assert agent.interactions == 1
    assert agent.query_counts['bootstrap']['queries'] == (6 if will_bootstrap else 0)
    assert agent.root_nodes['FIRE'].q_value == 6.


def test_missing_final_observation_and_bad_facts_do_not_update():
    agent = pong_agent()
    invalid = [
        ([], None, False, True),
        ([], None, False, False),
        ([Fact('logical', True, 'unknown', 'ball_t')], [], True, False),
        ([Fact('logical', True, 'present', 'ball_t'),
          Fact('logical', False, 'present', 'ball_t')], [], True, False),
    ]
    before = pickle.dumps(agent.state_dict())
    for state, following, terminal, truncated in invalid:
        with pytest.raises(ValueError):
            agent.observe(state, 0, 1., following, terminal, truncated)
        assert pickle.dumps(agent.state_dict()) == before


def test_serialized_learner_restores_trees_statistics_rng_and_future_updates():
    agent = pong_agent(eta_q=1., gamma=0., min_sample_size=6, significance_level=.05)
    for i in range(12):
        state, reward = signal(i)
        agent.observe(state, 0, reward, None, True, False)
        agent.select_action(state, epsilon=.5)
    assert agent.n_splits == 1
    saved = pickle.loads(pickle.dumps(agent.state_dict()))
    restored = RelationalBaselineAgent(agent.actions, agent.encoder, agent.config)
    restored.load_state_dict(saved)
    assert restored.contract() == saved['contract']
    saved['roots']['NOOP'].q_value = 1000.
    assert restored.root_nodes['NOOP'].q_value != 1000.
    for i in range(12, 120):
        state, reward = signal(i)
        assert restored.select_action(state, epsilon=.5) == agent.select_action(state, epsilon=.5)
        assert restored.observe(state, 0, reward, None, True, False) == (
            agent.observe(state, 0, reward, None, True, False))
    assert restored.n_splits == agent.n_splits == 3
    assert restored.interactions == agent.interactions == 120
    assert restored.query_counts == agent.query_counts
    assert restored.policy_rng.bit_generator.state == agent.policy_rng.bit_generator.state
    for action in agent.actions:
        assert tree_state(restored.root_nodes[action]) == tree_state(agent.root_nodes[action])


def test_catalog_order_hash_and_mismatched_restore():
    encoder = PongRelationsV1().encoder()
    agent = RelationalBaselineAgent(PONG_ACTIONS, encoder, ordered_catalog=encoder.catalog)
    reversed_agent = RelationalBaselineAgent(PONG_ACTIONS, encoder,
                                             ordered_catalog=tuple(reversed(encoder.catalog)))
    assert agent.contract()['ordered_literals_hash'] != reversed_agent.contract()['ordered_literals_hash']
    assert agent.contract()['ordered_catalog'] == [asdict(k) for k in encoder.catalog]
    before = pickle.dumps(agent.state_dict())
    with pytest.raises(ValueError, match='mismatch'):
        agent.load_state_dict(reversed_agent.state_dict())
    assert pickle.dumps(agent.state_dict()) == before
    changed = RelationalBaselineAgent(PONG_ACTIONS, encoder, replace(agent.config, eta_q=.5))
    with pytest.raises(ValueError, match='mismatch'):
        changed.load_state_dict(agent.state_dict())
    broken = agent.state_dict()
    broken['query_counts']['prediction']['internal_fallbacks'] = 1
    with pytest.raises(ValueError, match='query counters'):
        agent.load_state_dict(broken)
    assert pickle.dumps(agent.state_dict()) == before


def test_visual_catalog_uses_full_support_instead_of_historical_restriction(tmp_path):
    historical = CorrectedRRLAgent(BaselineConfig(game='Pong'), tmp_path)
    encoder = MultiGameRelations('Pong').encoder()
    agent = RelationalBaselineAgent(PONG_ACTIONS, encoder)
    assert len(historical.refinements) == 14
    assert len(agent.literals) == 24
    historical_catalog = {RelationKey(lit.type, lit.name, lit.obj1, lit.obj2, lit.obj3)
                          for lit in historical.refinements}
    assert historical_catalog < set(agent.catalog)
    assert agent.catalog == encoder.catalog
    assert agent.contract()['catalog_source'] == 'environment_encoder'
    assert RelationKey('comparative', 'y', 'player_t', 'player_t-1') in agent.catalog


def test_rejects_catalog_unknown_duplicates_and_ambiguous_historical_lookup():
    encoder = PongRelationsV1().encoder()
    for catalog in ((encoder.catalog[0],) * 2, (RelationKey('logical', 'unknown', 'ball_t'),)):
        with pytest.raises(ValueError, match='candidate catalog'):
            RelationalBaselineAgent(PONG_ACTIONS, encoder, ordered_catalog=catalog)
    ambiguous = RelationalEncoder([RelationKey('logical', 'present', 'ball_t'),
                                    RelationKey('logical', 'present', 'ball_t', obj3='enemy_t')])
    with pytest.raises(ValueError, match='ambiguous'):
        RelationalBaselineAgent(PONG_ACTIONS, ambiguous)


def test_selection_explicit_rng_uniform_ties_and_no_posterior():
    agent = pong_agent()
    before = copy.deepcopy(agent.policy_rng.bit_generator.state)
    rng = np.random.default_rng(20)
    choices = [agent.select_action([], rng=rng) for _ in range(120)]
    assert set(choices) == set(range(6))
    assert agent.policy_rng.bit_generator.state == before
    for epsilon in (-1., 1.01, float('nan')):
        with pytest.raises(ValueError):
            agent.select_action([], epsilon=epsilon)
    for method in (agent.predict_posterior, agent.predict_draws):
        with pytest.raises(NotImplementedError, match='no posterior'):
            method([])
