import copy
from dataclasses import replace
import pickle

import numpy as np
import pytest

from bayesian_rrtl.baseline_adapter import RelationalBaselineAgent
from bayesian_rrtl.stable_revision import StableRevisionAgent, variant_config, return_gate
from bayesian_rrtl.tree_revision import RevisionAgent, RevisionConfig, walk
from test_tree_revision import base, planted, state, transition


def fork(name='baseline', parent=None, **options):
    return StableRevisionAgent.fork(parent or base(),
        RevisionConfig(min_train=8,min_validation=8,min_validation_blocks=4,min_child=2,
                       max_candidates=2,cooldown=4,bootstrap_samples=100),
        variant_config(name,**options))


@pytest.mark.parametrize('criterion',['historical_f','anova'])
def test_baseline_hook_is_exact_with_growth_terminal_and_truncation(criterion):
    template=base(criterion)
    original=RelationalBaselineAgent(template.actions,template.encoder,replace(template.config,eta_q=1.,gamma=0.))
    old=RevisionAgent.fork(original,RevisionConfig())
    new=StableRevisionAgent.fork(original,RevisionConfig(),variant_config('baseline',eta_q=1.))
    for i in range(180):
        s=state('less' if i%2 else 'more');ns=state('more' if i%2 else 'less')
        args=(s,2,float(i%2),ns,i%11==0,i%13==0)
        assert old.observe(*args)==new.observe(*args)
        np.testing.assert_array_equal(old.predict_q(s),new.predict_q(s))
        assert old.n_splits==new.n_splits
    assert new.grows_total>0
    new.validate_revision()


def test_double_selection_uses_online_and_evaluation_uses_frozen_target():
    a=fork('double_td',target_interval=3)
    for root in a.root_nodes.values():root.q_value=0.
    for root in a.target_roots.values():root.q_value=0.
    a.root_nodes['FIRE'].q_value=5.
    a.target_roots['NOOP'].q_value=10.
    a.target_roots['FIRE'].q_value=2.
    assert a._bootstrap_value(tuple(state()))==2.
    td=a.observe(state(),2,0.,state(),False,False)
    assert td.target==a.config.gamma*2.
    assert a.target_roots['NOOP'].q_value==10. and a.target_counter==0
    a.observe(state(),2,0.,state(),False,False)
    assert a.target_syncs==0
    a.observe(state(),2,0.,state(),False,False)
    assert a.target_syncs==1 and a.target_counter==3
    assert a.target_roots['NOOP'] is not a.root_nodes['NOOP']
    a.validate_revision()


def test_double_ties_are_averaged_without_changing_policy_rng():
    a=fork('double_td')
    for root in a.root_nodes.values():root.q_value=1.
    for i,root in enumerate(a.target_roots.values()):root.q_value=float(i)
    before=pickle.dumps(a.policy_rng.bit_generator.state)
    assert a._bootstrap_value(tuple(state()))==1.5
    assert pickle.dumps(a.policy_rng.bit_generator.state)==before


def test_terminal_does_not_query_target_and_truncation_bootstraps():
    a=fork('double_td');a._bootstrap_value=lambda st: (_ for _ in ()).throw(AssertionError('queried'))
    assert a.observe(state(),2,1.,None,True,False).target==1.
    b=fork('double_td')
    for root in b.target_roots.values():root.q_value=2.
    assert b.observe(state(),2,1.,state(),False,True).target==1.+2*b.config.gamma


def test_projection_prevents_target_refit_leaving_old_prediction_hull():
    a=fork('guarded',planted())
    old=a.root_nodes['RIGHT'];old.more_branch.q_value=2.;old.less_branch.q_value=0.
    rows=[transition(i,99.) for i in range(20)]
    candidate=a._bounded(old,a._new_node(old),rows,np.full(20,99.))
    assert candidate.q_value==2.  # Neither target99 nor stale internal Q99.


def test_policy_trust_rejects_improved_loss_before_rollout_gate():
    a=fork('guarded',planted())
    for action,root in a.root_nodes.items():
        if action!='RIGHT':root.q_value=.5
    root=a.root_nodes['RIGHT'];root.less_branch.q_value=0.;root.more_branch.q_value=1.
    a.buffer.extend(transition(i,.6) for i in range(80))
    def forbidden(*args):raise AssertionError('return gate should not be reached')
    before=list(a.buffer);results=a.maintain([transition(i,.6) for i in range(80,160)],forbidden)
    assert results and any(r['loss_gate']['accepted'] for r in results)
    assert all(not r['gate']['accepted'] for r in results)
    assert any(r['gate']['reason']=='policy_trust' for r in results)
    assert a.prunes_total==a.replacements_total==0 and list(a.buffer)==before and a.interactions==0


def test_guarded_real_prune_requires_return_acceptance_and_does_not_train_on_holdout():
    a=fork('guarded',planted());a.buffer.extend(transition(i) for i in range(40))
    valid=[transition(i) for i in range(40,80)];before=list(a.buffer)
    rejected=a.maintain(valid,lambda *args:{'accepted':False,'reason':'synthetic_return_drop'})
    assert rejected and a.prunes_total==0
    accepted=a.maintain(valid,lambda *args:{'accepted':True,'reason':'synthetic_nondegradation'})
    assert accepted[-1]['gate']['accepted'] and a.prunes_total==1
    assert a.root_nodes['RIGHT'].literal is None
    assert list(a.buffer)==before and a.interactions==0
    a.validate_revision()


def test_guarded_missing_callback_cannot_silently_skip_return_validation():
    with pytest.raises(ValueError,match='rollout-return'):fork('guarded',planted()).maintain([transition(0)])


def test_protected_replacement_installs_alternative_partition_within_old_hull():
    a=fork('guarded',planted())
    for action,root in a.root_nodes.items():
        if action!='RIGHT':root.q_value=10.  # The repair preserves the greedy policy.
    root=a.root_nodes['RIGHT'];old=root.literal
    root.less_branch.q_value=0.;root.more_branch.q_value=1.
    a.buffer.extend(transition(i,float(i%2)) for i in range(80))
    results=a.maintain([transition(i,float(i%2)) for i in range(80,160)],lambda *args:{'accepted':True})
    assert results[-1]['gate']['accepted'] and results[-1]['kind']=='replacement_review'
    assert a.replacements_total==1 and a.prunes_total==0
    root=a.root_nodes['RIGHT']
    assert root.literal!=old and root.literal.obj2=='ball_t'
    assert root.predict(state('less'))==1. and root.predict(state('more'))==0.
    a.validate_revision()


def test_refit_changes_only_values_not_topology_statistics_or_split_counter():
    a=fork('refit_only',planted());a.buffer.extend(transition(i) for i in range(40))
    original=[(p,n.revision_uid,n.literal,copy.deepcopy(n.refinements_stats)) for p,n in walk(a.root_nodes['RIGHT'])]
    results=a.maintain([transition(i) for i in range(40,80)])
    assert results[-1]['gate']['accepted'] and a.refits_total==1
    assert a.n_splits==1 and a.structure_version==0 and a.parameter_version==1
    assert a.root_nodes['RIGHT'].predict(state(motion='more'))==1.
    assert a.root_nodes['RIGHT'].predict(state(motion='same'))==0.  # Unobserved route is preserved.
    assert [(p,n.revision_uid,n.literal,n.refinements_stats) for p,n in walk(a.root_nodes['RIGHT'])]==original
    a.validate_revision()


def test_paired_return_guard_checks_mean_lower_bound_and_individual_drops():
    cfg=variant_config('guarded',return_margin=1000.)
    assert return_gate([1,2,3],[2,3,4],cfg,np.random.default_rng(3))['accepted']
    assert not return_gate([1,2],[.9,1.9],cfg,np.random.default_rng(3))['accepted']
    assert not return_gate([100,1],[1,150],cfg,np.random.default_rng(3))['accepted']
    with pytest.raises(ValueError):return_gate([1,float('nan')],[2,3],cfg,np.random.default_rng(3))


def test_pickle_resume_preserves_target_cadence_and_all_updates():
    a=fork('stable_td',target_interval=7)
    for i in range(12):a.observe(state(),2,float(i%2),state(),False,False)
    b=pickle.loads(pickle.dumps(a))
    for i in range(60):
        args=(state('less' if i%2 else 'more'),2,float(i%2),state(),i%11==0,i%13==0)
        assert a.observe(*args)==b.observe(*args)
    assert a.tree_export()==b.tree_export() and a.target_counter==b.target_counter and a.events==b.events
    a.validate_revision();b.validate_revision()
