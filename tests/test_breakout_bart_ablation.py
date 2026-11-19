import json
import sys
from types import SimpleNamespace

import numpy as np

from bayesian_rrtl.ensemble import BARTRegressor
from bayesian_rrtl.online.learners import LearningConfig, OnlineBatchAgent
from bayesian_rrtl.tree import Tree
from experiments import breakout_bart_ablation
from experiments.breakout_bart_ablation import ServiceController, service_summary
from test_bayesian_kernel import tiny_data


def test_warm_start_reuses_supported_trees_and_resets_unsupported_ones():
    batch, y, rules = tiny_data()
    regressor = BARTRegressor(m=2)
    prior = regressor.tree_prior
    eligible = prior.eligible(rules, batch, np.arange(len(batch)), 0)
    rule = next(iter(eligible.values()))[0]
    initial = Tree(rule=rule, true_child=Tree(0.1), false_child=Tree(-0.1))
    rng = np.random.default_rng(14)
    supported = regressor.fit(batch, y, rng, burn_in=0, draws=4, initial_trees=(initial, Tree()))
    assert supported.diagnostics()['reused_trees'] == 2
    constant = batch.take([0] * len(batch))
    unsupported = regressor.fit(constant, y, rng, burn_in=0, draws=4, initial_trees=(initial, Tree()))
    assert unsupported.diagnostics()['reused_trees'] == 1
    assert unsupported.diagnostics()['reset_trees'] == 1
    assert all(t.is_leaf for draw in unsupported.draws for t in draw.trees)
    assert initial.true_child.mu == 0.1


def test_cadence_variant_anchors_first_fit_at_warmup():
    config = LearningConfig(('FIRE', 'RIGHT'), seed=1, warmup_interactions=512, fit_interval=1024)
    agent = SimpleNamespace(config=config, last_fit_step=0, learner_id='bart',
                            estimator_config={'first_fit_at_warmup': True})
    assert not OnlineBatchAgent.fit_due(agent, 511)
    assert OnlineBatchAgent.fit_due(agent, 512)
    agent.last_fit_step = 512
    assert not OnlineBatchAgent.fit_due(agent, 1024)
    assert OnlineBatchAgent.fit_due(agent, 1536)
    agent.last_fit_step = 0
    agent.estimator_config = {}
    assert not OnlineBatchAgent.fit_due(agent, 512)
    assert OnlineBatchAgent.fit_due(agent, 1024)


def test_array_pairs_variants_within_seed(monkeypatch, tmp_path):
    config = tmp_path / 'config.json'
    config.write_text(json.dumps({'variants': {'control': {}, 'interval_1024': {}},
                                  'training_seeds': [1, 2]}))
    calls = []
    monkeypatch.setattr(breakout_bart_ablation, 'run_cell',
                        lambda _, variant, index, seconds: calls.append((variant, index, seconds)))
    for task in range(4):
        monkeypatch.setattr(sys, 'argv', ['experiment', 'run', '--output', str(tmp_path),
                                         '--config', str(config), '--task', str(task)])
        breakout_bart_ablation.main()
    assert calls == [('control', 0, 14280), ('interval_1024', 0, 14280),
                     ('control', 1, 14280), ('interval_1024', 1, 14280)]


def test_fire_delays_include_unsuccessful_service_opportunities():
    agent = SimpleNamespace(model=None, encoder=None, config=SimpleNamespace(actions=('NOOP', 'FIRE')))
    controller = ServiceController(agent)
    env = SimpleNamespace(steps=0, last_auxiliary={'life_lost': False})
    for step, action, lost, truncated in ((1, 0, False, False), (2, 1, False, False),
                                          (8, 0, True, False), (15, 0, False, True)):
        env.steps = step
        env.last_auxiliary['life_lost'] = lost
        controller.transition(None, action, None, 0, 0, False, truncated, env)
    result = [{'transitions': controller.records}]
    assert service_summary(result, 'reset')['mean_delay_given_fire'] == 2
    assert service_summary(result, 'life_lost')['fires'] == 0
    assert service_summary(result, 'life_lost')['censored_before_32'] == 1
