"""ANOVA growth references, updated-Q semantics and checkpoint continuation."""
import copy
from dataclasses import asdict, replace
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from scipy.stats import f_oneway

from bayesian_rrtl.baseline_anova import ANOVANode, anova_from_stats
from bayesian_rrtl.baseline_adapter import BaselineLearnerConfig, RelationalBaselineAgent
from bayesian_rrtl.baseline_checkpoint import _validate_trees
from bayesian_rrtl.baseline_training import BaselineTrainingSession
from bayesian_rrtl.config import BaselineConfig
from bayesian_rrtl.online.protocol import dry_run, resolve
from experiments.validate_baseline_resume import SplitSignalEnvironment, continuation
from preprocessing import Fact
from relational_regresion_tree import Literal

ROOT = Path(__file__).resolve().parents[1]


def moments(values):
    n, mu, j = 0, 0., 0.
    for value in values:
        n += 1
        old = mu
        mu += (value-mu)/n
        j += (value-old)*(value-mu)
    return {'n': n, 'mu': mu, 'J': j}


def grouped(values):
    return {'Total': moments([v for group in values for v in group]),
            **{str(i): moments(group) for i, group in enumerate(values)}}


@pytest.mark.parametrize('sizes', [(9, 11), (6, 8, 13), (1, 3, 8), (0, 6, 9), (7, 0, 12)])
def test_anova_matches_scipy_oneway(sizes):
    rng = np.random.default_rng(53)
    groups = [rng.normal(i*.7, 1+i*.2, size=n).tolist() for i, n in enumerate(sizes)]
    expected = f_oneway(*(g for g in groups if g))
    result = anova_from_stats(grouped(groups))
    assert result.statistic == pytest.approx(expected.statistic, rel=1e-12, abs=1e-14)
    assert result.p_value == pytest.approx(expected.pvalue, rel=1e-12, abs=1e-14)
    assert result.k == sum(bool(g) for g in groups) and result.n == sum(sizes)


@pytest.mark.parametrize('groups,reason,p_value', [
    ([[], []], 'fewer_than_two_groups', None),
    ([[1., 2., 3.], []], 'fewer_than_two_groups', None),
    ([[0.], [2.]], 'no_residual_degrees_of_freedom', None),
    ([[2., 2.], [2., 2.]], 'zero_total_dispersion', None),
    ([[0., 0., 0.], [2., 2.]], None, 0.),
    ([[-1., 1.], [-2., 2.]], None, 1.),
])
def test_degenerate_and_equal_mean_cases(groups, reason, p_value):
    result = anova_from_stats(grouped(groups))
    assert result.reason == reason and result.p_value == p_value


@pytest.mark.parametrize('change', ['negative_J', 'nan', 'infinity', 'counts', 'pooled_J', 'pooled_mean', 'empty_J'])
def test_invalid_moments_raise_instead_of_becoming_no_split(change):
    stats = grouped([[1., 2., 4.], [5., 7.]])
    if change == 'negative_J':
        stats['0']['J'] = -1e-20
    elif change == 'nan':
        stats['0']['mu'] = float('nan')
    elif change == 'infinity':
        stats['1']['J'] = float('inf')
    elif change == 'counts':
        stats['Total']['n'] += 1
    elif change == 'pooled_J':
        stats['Total']['J'] *= 2
    elif change == 'pooled_mean':
        stats['Total']['mu'] += 1
    else:
        stats['empty'] = {'n': 0, 'mu': 0., 'J': 1.}
    with pytest.raises(ValueError, match='ANOVA'):
        anova_from_stats(stats)


def test_numeric_extremes_and_invalid_survival_function(monkeypatch):
    stats = {'Total': {'n': 4, 'mu': 0., 'J': 4e300},
             'a': {'n': 2, 'mu': -1e150, 'J': 1e-300},
             'b': {'n': 2, 'mu': 1e150, 'J': 1e-300}}
    result = anova_from_stats(stats)
    assert result.p_value == 0. and result.statistic == float('inf')
    with pytest.raises(ValueError, match='pooled'):
        anova_from_stats({'Total': {'n': 4, 'mu': 0., 'J': 0.},
                          'a': {'n': 2, 'mu': 0., 'J': 1e-20},
                          'b': {'n': 2, 'mu': 0., 'J': 1e-20}})
    from bayesian_rrtl import baseline_anova
    monkeypatch.setattr(baseline_anova.f, 'sf', lambda *args: float('nan'))
    with pytest.raises(FloatingPointError, match='invalid p-value'):
        anova_from_stats(grouped([[1., 2., 4.], [5., 7.]]))


@pytest.mark.parametrize('config_type', [BaselineLearnerConfig, BaselineConfig])
def test_config_default_validation_and_serialization(config_type):
    assert config_type().split_test == 'historical_f'
    assert asdict(config_type(split_test='anova'))['split_test'] == 'anova'
    for value in ('F', None, True, [], {}):
        with pytest.raises(ValueError, match='split_test'):
            config_type(split_test=value)
    for criterion in ('f-ratio', 'max-q-value'):
        with pytest.raises(ValueError, match='p-value'):
            config_type(split_test='anova', best_literal_criteria=criterion)


def test_omitted_option_is_identical_to_explicit_historical_f():
    from experiments.validate_baseline_resume import tree_data
    env = SplitSignalEnvironment()
    options = dict(eta_q=1., gamma=0., min_sample_size=6, significance_level=.05)
    default = RelationalBaselineAgent(env.actions, env.encoder(), BaselineLearnerConfig(**options))
    explicit = RelationalBaselineAgent(env.actions, env.encoder(), BaselineLearnerConfig(
        **options, split_test='historical_f'))
    assert default.contract() == explicit.contract()
    state = env.reset(seed=0)
    for _ in range(120):
        following, reward, _, terminal, truncated = env.step(0)
        assert default.observe(state, 0, reward, following, terminal, truncated) == explicit.observe(
            state, 0, reward, following, terminal, truncated)
        assert tree_data(default.root_nodes['NOOP']) == tree_data(explicit.root_nodes['NOOP'])
        state = following
    assert default.n_splits == explicit.n_splits == 3


def test_select_smallest_p_value_and_inclusive_threshold():
    literals = [Literal(axis, 'comparative', 'player_t', 'ball_t') for axis in ('x', 'y')]
    node = ANOVANode('NOOP', refinements=literals, min_sample_size=1, valid_actions=['NOOP'])
    for literal, samples in zip(literals, [([0., 1.], [1., 2.]), ([0., 1.], [4., 5.])]):
        values = grouped(samples)
        node.refinements_stats[literal] = {'Total': values['Total'], 'more': values['0'],
            'less': values['1'], 'same': moments([])}
    assert node.find_best_literal() == literals[1]
    node.significance_level = node.anova_test(literals[1]).p_value
    node.q_value = 3.5
    state = [Fact('comparative', 'more', axis, 'player_t', 'ball_t') for axis in ('x', 'y')]
    assert node.split_iteration(state, inherit_q_values=True)
    assert node.literal == literals[1] and len(node.children) == 3
    assert all(type(child) is ANOVANode and child.q_value == 3.5 for child in node.children)
    assert [child.node_type for child in node.children] == ['more', 'same', 'less']
    assert all(child.refinements == [literals[0]] for child in node.children)


@pytest.mark.parametrize('inherit', [False, True])
def test_descendants_keep_anova_and_historical_routing_limits(inherit):
    env = SplitSignalEnvironment()
    agent = RelationalBaselineAgent(env.actions, env.encoder(), BaselineLearnerConfig(
        split_test='anova', eta_q=1., gamma=0., min_sample_size=6,
        significance_level=.05, max_depth=2, inherit_q_values=inherit))
    state = env.reset(seed=0)
    for _ in range(80):
        following, reward, _, terminal, truncated = env.step(0)
        leaf = agent.root_nodes['NOOP'].get_state_leaf(state)
        before = agent.n_splits
        td = agent.observe(state, 0, reward, following, terminal, truncated)
        if agent.n_splits > before:
            assert all(type(child) is ANOVANode for child in leaf.children)
            assert all(child.q_value == (td.new_q if inherit else 0.) for child in leaf.children)
        state = following
    assert agent.n_splits == 3
    _validate_trees(agent)
    root = agent.root_nodes['NOOP']
    assert agent.predict_q([])[0] == root.q_value  # Missing internal literal fallback.
    def visit(node):
        assert type(node) is ANOVANode and node.depth <= 2
        for child in node.children:
            assert child.parent is node
            visit(child)
    visit(root)
    for minimum, depth in ((1000, 2), (6, 0)):
        limited = RelationalBaselineAgent(env.actions, env.encoder(), replace(agent.config,
            min_sample_size=minimum, max_depth=depth))
        for i in range(40):
            state = [Fact('comparative', ('less', 'more')[i%2], 'x', 'player_t', 'ball_t')]
            limited.observe(state, 0, float(i%2), None, True, False)
        assert limited.n_splits == 0


def test_logical_branches_and_strict_minimum():
    literal = Literal('contact', 'logical', 'player_t', 'ball_t')
    node = ANOVANode('NOOP', refinements=[literal], min_sample_size=4, valid_actions=['NOOP'])
    values = grouped([[0., 0.], [2., 2.]])
    node.refinements_stats[literal] = {'Total': values['Total'], 'True': values['0'], 'False': values['1']}
    assert node.find_best_literal() is None  # n == min_sample_size still insufficient.
    node.min_sample_size = 3
    assert node.split_iteration([Fact('logical', 'True', 'contact', 'player_t', 'ball_t')], False)
    assert node.children == [node.true_branch, node.false_branch]
    assert all(child.q_value == 0. and type(child) is ANOVANode for child in node.children)


def test_statistics_are_updated_q_not_td_targets_and_state_contract_rejects_switch():
    env = SplitSignalEnvironment()
    config = BaselineLearnerConfig(split_test='anova', eta_q=.25, gamma=.5, max_depth=0)
    agent = RelationalBaselineAgent(env.actions, env.encoder(), config)
    state = env.reset(seed=0)
    td = agent.observe(state, 0, 8., state, False, False)
    stats = agent.root_nodes['NOOP'].refinements_stats[agent.literals[0]]
    assert td.target == 8. and td.new_q == 2.
    assert stats['Total']['mu'] == 2. and stats['less']['mu'] == 2.
    saved = agent.state_dict()
    restored = RelationalBaselineAgent(env.actions, env.encoder(), config)
    restored.load_state_dict(saved)
    assert restored.contract()['config']['split_test'] == 'anova'
    historical = RelationalBaselineAgent(env.actions, env.encoder(), replace(config, split_test='historical_f'))
    with pytest.raises(ValueError, match='contract'):
        historical.load_state_dict(saved)


def test_checkpoint_continuation_in_fresh_process(tmp_path):
    env = SplitSignalEnvironment()
    config = BaselineLearnerConfig(split_test='anova', eta_q=1., gamma=0., min_sample_size=6,
                                   significance_level=.05, max_depth=3)
    session = BaselineTrainingSession(RelationalBaselineAgent(env.actions, env.encoder(), config), env,
                                      environment_factory=SplitSignalEnvironment)
    session.run(12)
    checkpoint = session.save(tmp_path/'anova.rrtlc')
    before = checkpoint.read_bytes()
    expected = json.loads(json.dumps(continuation(session, 96)))
    resumed = BaselineTrainingSession.load(checkpoint, environment_factory=SplitSignalEnvironment,
                                           expected_config=config)
    assert json.loads(json.dumps(continuation(resumed, 96))) == expected
    assert resumed.agent.n_splits == 3
    child = tmp_path/'child.json'
    # Existing worker always continues 128 steps, so use its same continuation.
    loaded = BaselineTrainingSession.load(checkpoint, environment_factory=SplitSignalEnvironment)
    expected_child = json.loads(json.dumps(continuation(loaded, 128)))
    subprocess.run([sys.executable, '-m', 'experiments.validate_baseline_resume', '--worker', str(checkpoint),
                    '--output', str(child), '--synthetic'], check=True, timeout=60, capture_output=True, text=True)
    assert json.loads(child.read_text()) == expected_child
    assert checkpoint.read_bytes() == before
    with pytest.raises(ValueError, match='config mismatch'):
        BaselineTrainingSession.load(checkpoint, expected_config=replace(config, split_test='historical_f'),
                                      environment_factory=SplitSignalEnvironment)


@pytest.mark.parametrize('size', ['smoke', 'long_review'])
def test_paired_configs_differ_only_in_name_and_growth_test(size, tmp_path):
    paths = [ROOT/'configs'/f'rrtl_split_tests_{size}_{mode}.json' for mode in ('historical_f', 'anova')]
    configs = [json.loads(path.read_text()) for path in paths]
    stripped = []
    for config, mode in zip(configs, ('historical_f', 'anova')):
        resolved = resolve(config)
        assert len(resolved['cells']) == (3 if size == 'smoke' else 15)
        assert all(c['estimator']['split_test'] == mode for c in resolved['cells'])
        d = dry_run(config)
        assert d['reserved_test_episodes_executed'] == 0
        assert d['estimates'] == []  # No made-up Linux cost evidence.
        if size == 'long_review':
            from bayesian_rrtl.online.protocol import freeze, authorize
            from experiments.compare_pong_progressive import train_cell
            folder = tmp_path/mode
            freeze(config, folder)
            with pytest.raises(ValueError, match='Authorize'):
                train_cell(folder, 0)
            with pytest.raises(ValueError, match='compatible measured Linux costs'):
                authorize(folder, 100000, 'Review only, no compatible measurements', game='Breakout')
        clean = copy.deepcopy(config)
        clean.pop('name')
        clean['learners']['corrected_common'].pop('split_test')
        stripped.append(clean)
    assert stripped[0] == stripped[1]


def test_paired_common_runner_resume_report_and_no_reserved_evaluation(tmp_path):
    # Reuse C's injected RAM driver with the three fixed game contracts,
    # without loading Atari or consuming real reserved seed sets.
    from test_atari_block_c import FakeFactory
    from bayesian_rrtl.online.environments import GameEnvironmentConfig
    from bayesian_rrtl.online.protocol import freeze, authorize
    from bayesian_rrtl.online.storage import binding, load_frontier
    from experiments.compare_pong_progressive import train_cell, evaluate_cell
    from experiments.pong_analysis_runtime import checked_read
    from experiments.report_rrtl_split_tests import report, paired_protocols
    paths = []
    manifests = []
    for mode in ('historical_f', 'anova'):
        config = json.loads((ROOT/'configs'/f'rrtl_split_tests_smoke_{mode}.json').read_text())
        folder = tmp_path/mode
        manifest = freeze(config, folder)
        paths.append(folder)
        manifests.append(manifest)
        for game in config['games']:
            authorize(folder, 64, 'Discardable synthetic ANOVA regression', game=game)
        for index, cell in enumerate(manifest['cells']):
            factory = FakeFactory(GameEnvironmentConfig(**cell['environment']))
            train_cell(folder, index, factory=factory, decision_limit=19)
            final = train_cell(folder, index, factory=factory)
            assert final['state'] == 'completed' and final['counter'] == 64
            assert final['metrics']['learner']['split_test'] == mode
            payload, _ = load_frontier(folder/cell['id'], final, binding(manifest, cell))
            from bayesian_rrtl.online.session import OnlineSession
            bad = copy.deepcopy(payload['session'])
            bad['collector']['agent'].config = replace(bad['collector']['agent'].config,
                split_test='historical_f' if mode == 'anova' else 'anova')
            with pytest.raises(ValueError, match='split_test/config'):
                OnlineSession.restore(bad, cell, factory())
            evaluate_cell(folder, index, factory=factory)
        assert not (folder/'final_test'/'usage.json').exists()
    result = report(*paths)
    assert result['completed'] and len(result['runs']) == 6
    for run in result['runs']:
        assert run['training_seconds'] > 0 and run['final_development'] is not None
        assert [point['decisions'] for point in run['learning_curve']] == [0, 32, 64]
        assert all({'splits', 'leaves', 'max_depth', 'training_active_seconds', 'mean_raw_return'} <= point.keys()
                   for point in run['learning_curve'])
    assert len(result['paired_contrasts']) == 3
    assert all(c['matched_training_seeds'] and not c['pending_training_seeds'] for c in result['paired_contrasts'])
    json.dumps(result, allow_nan=False)
    # Pairing must reject a hidden change to scientific or operating conditions.
    changed = copy.deepcopy(manifests[1])
    changed['protocol']['learning']['gamma'] = .5
    with pytest.raises(ValueError, match='differ beyond'):
        paired_protocols(manifests[0], changed)
    # Results were reconstructed without touching final training checkpoints.
    for folder, manifest in zip(paths, manifests):
        for cell in manifest['cells']:
            assert checked_read(folder/cell['id']/'status.json')['counter'] == 64
