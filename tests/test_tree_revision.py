"""Meaningful revision tests: exact control, valid mutations, holdout and resume."""
import copy
from dataclasses import replace
import pickle

import numpy as np
import pytest

from bayesian_rrtl.baseline_adapter import BaselineLearnerConfig, RelationalBaselineAgent
from bayesian_rrtl.encoding import RelationKey, RelationalEncoder
from bayesian_rrtl.tree_revision import (RevisionAgent, RevisionConfig, fact_value,
    paired_loss_gate, route, walk)
from preprocessing import Fact
from relational_regresion_tree import Literal


def state(x='less', motion='same', *, missing=False):
    facts = [Fact('comparative', motion, 'x', 'player_t', 'player_t-1')]
    if not missing: facts.append(Fact('comparative', x, 'x', 'player_t', 'ball_t'))
    return facts


def base(criterion='anova'):
    catalog = [RelationKey('comparative','x','player_t','ball_t'),
               RelationKey('comparative','x','player_t','player_t-1')]
    return RelationalBaselineAgent(('NOOP','FIRE','RIGHT','LEFT'), RelationalEncoder(catalog),
        BaselineLearnerConfig(eta_q=.2, gamma=.5, max_depth=3, min_sample_size=6,
                             significance_level=.05, split_test=criterion))


@pytest.mark.parametrize('criterion', ['historical_f','anova'])
def test_normal_is_exact_incremental_control_with_growth(criterion):
    template = base(criterion)
    original = RelationalBaselineAgent(template.actions, template.encoder,
        replace(template.config, eta_q=1., gamma=0.))
    fork = RevisionAgent.fork(original, RevisionConfig())
    for i in range(160):
        s = state('less' if i % 2 else 'more', 'less' if i % 3 else 'more')
        ns = state('more' if i % 2 else 'less')
        terminal, truncated = i % 11 == 0, i % 13 == 0
        reward = float(i % 2)
        assert original.observe(s,0,reward,ns,terminal,truncated) == fork.observe(s,0,reward,ns,terminal,truncated)
        for st in [s,ns,state(missing=True)]:
            np.testing.assert_array_equal(original.predict_q(st),fork.predict_q(st))
        assert original.n_splits == fork.n_splits
    assert fork.grows_total > 0
    fork.validate_revision()


def planted(mode='prune_regrow'):
    agent = RevisionAgent.fork(base(), RevisionConfig(mode=mode, min_train=8,
        min_validation=8, min_validation_blocks=4, min_child=2,
        replacement_gain=.001, cooldown=4, bootstrap_samples=100))
    root = agent.root_nodes['RIGHT']
    lit = root.refinements[1]  # Previous paddle motion; deliberately wrong partition.
    root.literal = lit
    for label in ('more','same','less'):
        child = agent._new_node(root, refinements=[r for r in root.refinements if r != lit])
        child.parent, child.depth, child.node_type = root, 1, label
        child.q_value = 0.
        setattr(root,label+'_branch',child); root.children.append(child)
    root.q_value = 99.  # Deliberately stale parent; must NOT be used for collapse Q.
    agent._learner.n_splits = 1; agent.initial_splits = 1
    agent._assign_uids()
    return agent


def transition(i, target=1.):
    return {'state': tuple(state('less' if i%2 else 'more', 'less' if i%3 else 'more')),
            'next_state': (), 'action': 2, 'reward': target, 'terminated': True,
            'truncated': False, 'block': (i//2)%4}


def test_pruning_fits_recent_q_preserves_candidates_and_valid_checkpoint():
    agent = planted('prune'); agent.buffer.extend(transition(i) for i in range(32))
    result = agent.maintain([transition(i) for i in range(32,64)])
    assert result[0]['gate']['accepted']
    root = agent.root_nodes['RIGHT']
    assert root.literal is None and root.q_value == 1. and agent.n_splits == 0
    assert root.refinements == list(agent.literals)
    assert not root.children and agent.prunes_total == 1
    # Missing literals use the fresh value too, not the stale 99.
    assert root.predict(state(missing=True)) == 1.
    agent.validate_revision()
    restored = pickle.loads(pickle.dumps(agent)); restored.validate_revision()
    for i in range(20):
        restored.observe(state(),2,1.,None,True,False)
    assert restored.n_splits == 0  # Pure pruning never regrows.


def test_local_replacement_uses_different_root_and_learns_position():
    agent = planted('replace')
    train = [transition(i, float(i%2)) for i in range(80)]
    valid = [transition(i, float(i%2)) for i in range(80,160)]
    agent.buffer.extend(train)
    old = agent.root_nodes['RIGHT'].literal
    results = agent.maintain(valid)
    assert results[0]['gate']['accepted']
    assert agent.replacements_total == 1 and agent.prunes_total == 0
    root = agent.root_nodes['RIGHT']
    assert root.literal != old and root.literal.obj2 == 'ball_t'
    assert root.predict(state('less')) == 1. and root.predict(state('more')) == 0.
    agent.validate_revision()


def test_validation_never_enters_replay_or_updates_td():
    agent = planted(); agent.buffer.extend(transition(i) for i in range(16))
    before = list(agent.buffer); counter = agent.interactions
    agent.maintain([transition(i) for i in range(32,64)])
    assert list(agent.buffer) == before and agent.interactions == counter


def test_insufficient_blocks_and_harmful_candidate_rejected():
    cfg = RevisionConfig(min_validation=8, min_validation_blocks=4, bootstrap_samples=100)
    rng = np.random.default_rng(1)
    assert not paired_loss_gate(np.zeros(8), np.zeros(8), cfg,rng)['accepted']
    assert not paired_loss_gate(np.ones(8), np.arange(8)%4,cfg,rng)['accepted']
    assert paired_loss_gate(-np.ones(8),np.arange(8)%4,cfg,rng)['accepted']


def test_freeze_still_updates_q_and_audits_distinct_signals():
    agent = RevisionAgent.fork(base(), RevisionConfig(mode='frozen'))
    td = agent.observe(state(),2,1.,None,True,False)
    assert td.target == 1. and td.new_q == .2
    for i in range(80): agent.observe(state('more' if i%2 else 'less'),2,float(i%2),None,True,False)
    assert agent.n_splits == 0 and agent.interactions == 81
    report = agent.audit_report()
    assert report and all(set(r['signals']) == {'updated_q','target','td_error'} for r in report)
    assert any(r['signals']['updated_q']['groups'] != r['signals']['target']['groups'] for r in report)


def test_resume_is_exact_after_grow_and_prune():
    a = planted(); a.buffer.extend(transition(i) for i in range(32))
    a.maintain([transition(i) for i in range(32,64)])
    b = pickle.loads(pickle.dumps(a))
    for i in range(80):
        s = state('less' if i%2 else 'more')
        assert a.observe(s,2,float(i%2),None,True,False) == b.observe(s,2,float(i%2),None,True,False)
    assert a.tree_export() == b.tree_export()
    assert a.events == b.events and a.audit_report() == b.audit_report()
    assert a.n_splits > 0  # Regrowth actually occurs after cooldown.
    a.validate_revision(); b.validate_revision()
