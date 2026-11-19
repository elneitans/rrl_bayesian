import numpy as np
import pytest

from bayesian_rrtl.rrtl_followup import FollowupAgent, double_targets
from bayesian_rrtl.stable_revision import variant_config
from bayesian_rrtl.tree_revision import RevisionConfig, walk
from test_tree_revision import base, planted, state, transition


def test_frozen_slow_td_updates_values_without_splits():
    model = FollowupAgent.fork(base('anova'), RevisionConfig(mode='frozen'), variant_config('slow_td'))
    topology = [(node['action'], node['path'], node['literal']) for node in model.tree_export()]
    for i in range(180):
        model.observe(state(), 2, float(i % 2), state(), False, False)
    assert model.root_nodes['RIGHT'].q_value > 0
    assert topology == [(node['action'], node['path'], node['literal']) for node in model.tree_export()]
    assert model.grows_total == 0 and model.config.eta_q == .005
    model.validate_revision()


def test_partial_refit_uses_quarter_of_the_proposed_change_on_same_structure():
    revision = RevisionConfig(min_train=8, min_validation=8, min_validation_blocks=4, min_child=2)
    model = FollowupAgent.fork(planted(), revision, variant_config('refit_only'), refit_fraction=.25)
    node = model.root_nodes['RIGHT']
    records = [transition(i) for i in range(40)]
    targets = np.ones(40)
    full = FollowupAgent.fork(model, revision, variant_config('refit_only'))
    candidate = model._refit(node, records, targets)
    full_node = full.root_nodes['RIGHT']
    full_candidate = full._refit(full_node, records, targets)
    for (_, old), (_, damped), (_, fitted) in zip(walk(node), walk(candidate), walk(full_candidate)):
        assert damped.literal == old.literal
        assert damped.q_value == pytest.approx(old.q_value + .25 * (fitted.q_value - old.q_value))
    assert node.predict(state(motion='more')) == 0
    assert candidate.predict(state(motion='more')) == .25


def test_double_comparison_uses_identical_observations_and_respects_terminal():
    model, lagged = base(), base()
    for root in model.root_nodes.values():
        root.q_value = 0
    model.root_nodes['FIRE'].q_value = 5
    lagged.root_nodes['NOOP'].q_value = 10
    lagged.root_nodes['FIRE'].q_value = 2
    transitions = [
        {'terminated': False, 'truncated': False, 'next_state': state(), 'reward': 1},
        {'terminated': True, 'truncated': False, 'next_state': None, 'reward': 1},
    ]
    actual, terminal = list(double_targets(transitions, model.root_nodes, lagged.root_nodes,
                                           gamma=.9, bootstrap_on_truncation=True))
    assert actual == {'online_max': 5.5, 'frozen_max': 10., 'double': 2.8, 'mask': 1,
                      'online_greedy': ['FIRE'], 'frozen_greedy': ['NOOP']}
    assert terminal == {'online_max': 1, 'frozen_max': 1, 'double': 1, 'mask': 0}
