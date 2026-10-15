"""A3–A6 portable acceptance, including controlled synthetic inference."""
import importlib.util
import json
import os
from types import SimpleNamespace
import time

import numpy as np
import pytest

from bayesian_rrtl.encoding import RelationKey, RelationalEncoder
from bayesian_rrtl.objects import DetectedObject, ObjectFrame
from bayesian_rrtl.priors import FixedScale, NoisePrior, TreePrior
from experiments.audit_bart_fixed_data import observable_diagnostics, run_chain, summarize
from experiments.audit_pong_representation import audit_rows, collision_examples, target_stats
from experiments.pong_common_contract import (CommonPongConfig, CommonPongEnvironment,
    CommonPongFactory, installed_rom_hash, validate_contract)
from experiments.pong_dataset_selection import partition, select
from preprocessing import Fact


class Driver:
    actions = ('NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE')
    def __init__(self):
        self.steps = 0
        self.calls = []
        self.current_frame = None
    def frame(self):
        return ObjectFrame(tuple(DetectedObject(name, True, box) for name, box in (
            ('player', (140., 100.+self.steps, 4., 8.)), ('ball', (80., 100., 2., 2.)),
            ('enemy', (10., 100., 4., 8.)))))
    def reset(self, *, seed):
        self.steps = 0
        self.current_frame = self.frame()
        return self.current_frame
    def step(self, action):
        self.calls.append(action)
        self.steps += 1
        self.current_frame = self.frame()
        return SimpleNamespace(frame=self.current_frame, raw_reward=(0, 3, -2)[self.steps % 3],
                               terminated=False, truncated=False)
    def snapshot(self):
        return {'steps': self.steps, 'calls': self.calls.copy()}
    def restore(self, saved):
        self.steps, self.calls = saved['steps'], saved['calls'].copy()
        self.current_frame = self.frame()
    def close(self):
        pass


def test_common_dynamics_facts_reset_rewards_and_contract_rejection():
    visual, oc = Driver(), Driver()
    rgb = CommonPongEnvironment(CommonPongConfig('visual', '0'*64, 5), driver=visual)
    ram = CommonPongEnvironment(CommonPongConfig('ocatari', '0'*64, 5), driver=oc)
    assert rgb.reset(seed=600001) == ram.reset(seed=600001)
    assert not visual.calls and not oc.calls  # reset has no hidden FIRE
    for action in (1, 0, 3, 2, 4):
        left, right = rgb.step(action), ram.step(action)
        assert left == right
        assert left[1] == np.sign(left[2]) and left[1] in (-1, 0, 1)
        assert visual.snapshot() == oc.snapshot()  # identical evolution before extraction
    assert left[4] and not left[3]
    assert rgb.encoder().catalog_hash == ram.encoder().catalog_hash
    assert len(rgb.encoder().catalog) == 20
    with pytest.raises(ValueError, match='contract'):
        validate_contract(rgb.config.manifest(), {**rgb.config.manifest(), 'reward_transform': 'shaping'})
    snapshot = rgb.snapshot()
    changed = CommonPongEnvironment(CommonPongConfig('visual', '0'*64, 6), driver=Driver())
    with pytest.raises(ValueError, match='contract'):
        changed.restore(snapshot)


@pytest.mark.skipif(os.environ.get('RRL_COMMON_EMULATOR_TEST') != '1' or
                    importlib.util.find_spec('ocatari') is None,
                    reason='Opt-in original Linux OCAtari environment required')
def test_real_ale_dynamics_match_between_extractors():
    digest = installed_rom_hash()
    rgb = CommonPongFactory(CommonPongConfig('visual', digest, 64))()
    ram = CommonPongFactory(CommonPongConfig('ocatari', digest, 64))()
    try:
        rgb.reset(seed=600991)
        ram.reset(seed=600991)
        for action in (1, 0, 2, 3, 4, 5)*3:
            rgb.step(action)
            ram.step(action)
            np.testing.assert_array_equal(rgb.backend.env.unwrapped.ale.getRAM(),
                                          ram.backend.env._env.unwrapped.ale.getRAM())
    finally:
        rgb.close()
        ram.close()


def data_fixture():
    return {'transition_ids': np.arange(8), 'episode_ids': np.repeat(np.arange(4), 2),
        'step_ids': np.tile([0, 1], 4), 'actions': np.tile([0, 1], 4),
        'rewards': np.array([0, 1, 0, -1, 0, 0, 1, -1], dtype=float),
        'raw_rewards': np.array([0, 2, 0, -2, 0, 0, 2, -2], dtype=float),
        'terminated': np.array([False, True]*3+[False, False]), 'truncated': np.zeros(8, dtype=bool),
        'state_values': np.array([[0], [1], [0], [1], [-1], [1], [-1], [1]], dtype=np.int8),
        'state_observed': np.array([[True], [True], [True], [True], [False], [True], [False], [True]]),
        'next_values': np.array([[1], [0], [1], [0], [1], [0], [1], [0]], dtype=np.int8),
        'next_observed': np.ones((8, 1), dtype=bool)}


def test_complete_episode_split_and_deterministic_selection_no_duplicates():
    data = data_fixture()
    train, holdout, meta = partition(data, .33)
    assert meta['complete'] == [0, 1, 2] and meta['holdout'] == [2]
    assert meta['excluded_incomplete'] == [3]
    assert not np.any(train & holdout) and not (train | holdout)[6:].any()
    for rule in ('uniform_action', 'stratified_reward'):
        first, details = select(data, train, actions=2, per_action=128, seed=600701, rule=rule)
        second, _ = select(data, train, actions=2, per_action=128, seed=600701, rule=rule)
        np.testing.assert_array_equal(first, second)
        assert not np.any(first & holdout)
        assert len(set(data['transition_ids'][first])) == first.sum()
        assert details['coverage'][0]['no_positive_signal']
        assert details['distribution_changed'] == (rule == 'stratified_reward')


def test_portable_audit_exact_ids_unknown_causes_and_auxiliary():
    data = data_fixture()
    encoder = RelationalEncoder([RelationKey('logical', 'present', 'player_t')])
    replay = audit_rows(data, encoder, np.arange(8), convention='screen_y_down')
    subset = audit_rows(data, encoder, [1, 4, 5], convention='screen_y_down')
    assert sum(replay['by_action'].values()) == sum(replay['by_episode'].values()) == 8
    assert sum(replay['raw_rewards'].values()) == 8
    assert subset['exact_ids'] == [1, 4, 5] and subset['transitions'] == 3
    assert subset['relations'][0]['missing_causes'] == {'unknown': 1}
    data['relation_causes'] = np.array([['observed']]*4+[['observed_absence'], ['observed'], ['history_reset'], ['observed']])
    auxiliary = audit_rows(data, encoder, np.arange(8), convention='screen_y_down')
    assert auxiliary['relations'][0]['missing_causes'] == {'history_reset': 1, 'observed_absence': 1}
    data['geometry_json'] = np.array([json.dumps({'x': i}) for i in range(8)])
    assert collision_examples(data)['examples'][0]['geometry_different']
    assert target_stats([-2, 0, 2], -1, 1)['outside_declared_scale_fraction'] == 2/3


def collision_fixture(actions, next_values, geometry=None):
    n = len(actions)
    data = {'state_values': np.zeros((n, 1), dtype=np.int8),
            'state_observed': np.ones((n, 1), dtype=bool),
            'actions': np.array(actions), 'transition_ids': np.arange(n),
            'next_values': np.array(next_values).reshape(n, 1),
            'next_observed': np.ones((n, 1), dtype=bool),
            'raw_rewards': np.zeros(n)}
    if geometry is not None:
        data['geometry_json'] = np.array(geometry)
    return data


def test_audit_different_actions_do_not_imply_information_loss():
    data = collision_fixture([0, 1], [0, 1], ['{"x":10}', '{"x":10}'])
    result = collision_examples(data)
    assert result['examples'] == result['geometry_collisions'] == result['same_action_evolution_differences'] == []


def test_audit_conditions_evolution_on_action_beyond_first_state_representative():
    data = collision_fixture([0, 1, 1], [0, 0, 1], ['{"x":10}']*3)
    result = collision_examples(data)
    assert result['geometry_collisions'] == []
    example, = result['same_action_evolution_differences']
    assert example['ids'] == [1, 2] and example['actions'] == [1, 1]
    assert example['same_action'] and example['evolution_different'] and not example['geometry_different']
    assert 'do not by themselves establish aliasing' in result['interpretation']


def test_audit_geometry_collisions_are_separate_and_json_order_is_irrelevant():
    data = collision_fixture([0, 1, 1], [0, 1, 1],
        ['{"x":10,"y":20}', '{"y":20,"x":10}', '{"x":11,"y":20}'])
    result = collision_examples(data)
    geometry, = result['geometry_collisions']
    assert geometry['ids'] == [0, 2] and geometry['geometry_different']
    assert not geometry['same_action'] and geometry['evolution_different'] is None
    assert result['same_action_evolution_differences'] == []
    del data['geometry_json']
    assert collision_examples(data)['examples'] == []


def fixed_job():
    encoder = RelationalEncoder([RelationKey('logical', 'present', 'player_t')])
    batch = encoder.transform([[Fact('logical', bool(i % 2), 'present', 'player_t')] for i in range(12)])
    return {'batch': batch, 'targets': np.array([0., 1.]*6), 'reference': batch.take([0, 1]),
        'reference_rule': ['frequent', 'rare'], 'prior': TreePrior(max_depth=2, min_child=1),
        'noise': NoisePrior(), 'scale': FixedScale(-2., 2.), 'k': 2., 'm': 2,
        'data_hash': 'fixed', 'transition_ids_sha256': 'ids',
        'calibration_method': 'synthetic', 'variance_floor_used': False}


def test_mcmc_exact_resume_extension_and_no_repeat(tmp_path, monkeypatch):
    from bayesian_rrtl.ensemble import BARTRegressor
    job = fixed_job()
    options = {'chain_seed': 600301, 'burn_in': 2, 'draws': 8,
               'deadline': time.monotonic()+60, 'checkpoint_every': 1}
    assert run_chain(job, tmp_path/'whole', **options) == 'completed'
    original = BARTRegressor.sweep
    calls = 0
    def fail_after_two(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError('simulated interruption')
        return original(self, *args, **kwargs)
    monkeypatch.setattr(BARTRegressor, 'sweep', fail_after_two)
    with pytest.raises(RuntimeError, match='interruption'):
        run_chain(job, tmp_path/'resume', **options)
    monkeypatch.setattr(BARTRegressor, 'sweep', original)
    assert run_chain(job, tmp_path/'resume', **options) == 'completed'
    with np.load(tmp_path/'whole/trace.npz') as whole, np.load(tmp_path/'resume/trace.npz') as resumed:
        for name in whole.files:
            if name != 'sampling_seconds':
                np.testing.assert_array_equal(whole[name], resumed[name])
        for move in ('grow', 'prune', 'change', 'swap'):
            np.testing.assert_array_equal(whole[f'{move}_selected'], whole[f'{move}_impossible']+
                whole[f'{move}_accepted']+whole[f'{move}_rejected'])
    assert run_chain(job, tmp_path/'resume', extend_draws=5, extension_id='longer', **options) == 'completed'
    assert run_chain(job, tmp_path/'resume', extend_draws=5, extension_id='longer', **options) == 'completed'
    with np.load(tmp_path/'resume/trace.npz') as extended:
        assert len(extended['sigma2']) == 15
    def must_not_sweep(*args, **kwargs):
        raise AssertionError('completed work repeated')
    monkeypatch.setattr(BARTRegressor, 'sweep', must_not_sweep)
    assert run_chain(job, tmp_path/'resume', **options) == 'completed'
    with pytest.raises(ValueError, match='identity'):
        run_chain({**job, 'm': 3}, tmp_path/'resume', **options)


def test_constants_stuck_mcse_units_and_valid_initializations(tmp_path):
    constant = observable_diagnostics(np.ones((4, 8)), 1)
    assert constant['interpretation'] == 'nonconclusive_constant' and not constant['passes']
    known = observable_diagnostics(np.ones((4, 8)), 1, known_degenerate=True)
    assert known['interpretation'] == 'known_support_degeneracy'
    stuck = observable_diagnostics(np.repeat(np.arange(4)[:, None], 8, axis=1), 1)
    assert stuck['interpretation'] == 'chains_stuck_at_different_constants' and not stuck['passes']
    iid = observable_diagnostics(np.random.default_rng(600301).normal(size=(4, 2000)), 2)
    assert iid['rhat'] < 1.01 and iid['ess_bulk'] > 400 and iid['mcse_mean'] > 0
    job = fixed_job()
    for initialization in ('stumps', 'dispersed'):
        job['initialization'] = initialization
        folder = tmp_path/initialization
        for chain in range(4):
            assert run_chain(job, folder/f'chain_{chain}', chain_seed=600301+chain,
                             burn_in=5, draws=20, deadline=time.monotonic()+60) == 'completed'
        report = summarize(job, folder, 4, 5)
        assert report['sigma2_original_median'] == pytest.approx(report['sigma2_median']*16)
        assert report['diagnostic_groups']['predictive']['prediction_0']['mcse_mean'] is not None
        assert report['sampling_seconds'] > 0 and report['statistical_quality'] == 'insufficient'
        predictive = report['diagnostic_groups']['predictive']
        assert predictive['prediction_1']['mean'] > predictive['prediction_0']['mean']+.25


def test_fqi_iteration_reuse_and_data_identity(tmp_path, monkeypatch):
    from bayesian_rrtl.q_config import BayesianQConfig
    from bayesian_rrtl.replay import Transition
    from experiments.compare_fqi_trees import FittedQ, fit_iteration
    import experiments.compare_fqi_trees as fqi
    job = fixed_job()
    encoder = RelationalEncoder(job['batch'].catalog)
    transitions = tuple(Transition(i, job['batch'].take([i]), 0, float(i % 2), float(i % 2),
                                   None, True, False, i, 0) for i in range(12))
    config = BayesianQConfig(actions=('a',), gamma=.5, reward_min=0., reward_max=1., min_child=1)
    previous = FittedQ(('a',), (None,), encoder.catalog_hash)
    expected = {'hashes': {'previous_model': 'zero', 'targets': 'fixed'}, 'dataset': 'original'}
    model, details = fit_iteration(tmp_path/'fit', expected, 'cart', transitions, encoder,
                                   config.to_dict(), previous, 1)
    def must_not_fit(*args, **kwargs):
        raise AssertionError('completed FQI iteration repeated')
    monkeypatch.setattr(fqi, 'fit_epoch', must_not_fit)
    restored, restored_details = fit_iteration(tmp_path/'fit', expected, 'cart', transitions, encoder,
                                               config.to_dict(), previous, 1)
    np.testing.assert_array_equal(restored.predict(job['batch']), model.predict(job['batch']))
    assert restored_details == details
    assert len(details['target_hashes']) == 1
    with pytest.raises(ValueError, match='identity'):
        fit_iteration(tmp_path/'fit', {**expected, 'dataset': 'new'}, 'cart', transitions,
                      encoder, config.to_dict(), previous, 1)


def test_new_regression_generator_never_fits_holdout_and_estimators_share_xy(tmp_path, monkeypatch):
    from bayesian_rrtl.q_config import BayesianQConfig
    from experiments.freeze_pong_analysis_data import publish_portable
    from experiments.freeze_pong_regression import freeze
    from experiments.compare_fqi_trees import regression_benchmark
    from experiments.pong_analysis_runtime import bytes_hash
    import experiments.compare_fqi_trees as fqi
    import io
    data = data_fixture()
    train, holdout, split = partition(data, .33)
    selected, selection = select(data, train, actions=2, per_action=128, seed=600701)
    data.update(episode_train=train, episode_holdout=holdout, fqi_selected=selected,
                next_present=~data['terminated'])
    encoder = RelationalEncoder([RelationKey('logical', 'present', 'player_t')])
    config = BayesianQConfig(actions=('a', 'b'), gamma=.5, reward_min=-1., reward_max=1., min_child=1)
    stream = io.BytesIO()
    np.savez_compressed(stream, **data)
    meta = {'kind': 'fixed_transitions', 'npz_sha256': bytes_hash(stream.getvalue()),
        'common_protocol': {'contract': 'synthetic'}, 'encoder': encoder.to_dict(),
        'config': config.to_dict(), 'fqi_selected_count': int(selected.sum())}
    dataset = tmp_path/'dataset'
    publish_portable(dataset, meta, stream.getvalue())
    observed = []
    original = fqi.fit_action
    def record_generator(method, batch, y, cfg, seed, **kwargs):
        observed.append((batch.values.copy(), np.asarray(y).copy()))
        return original(method, batch, y, cfg, seed, **kwargs)
    monkeypatch.setattr(fqi, 'fit_action', record_generator)
    problem = tmp_path/'problem'
    frozen = freeze(dataset, problem, generator_epochs=2)
    assert frozen['generator_train_ids'] == [0, 1, 2, 3]
    assert len(observed) == 4 and all(len(y) == 2 for _, y in observed)
    repeated = freeze(dataset, problem, generator_epochs=2)
    assert repeated == frozen and len(observed) == 4
    model_inputs = {}
    def identical_inputs(method, batch, y, cfg, seed, **kwargs):
        model_inputs.setdefault(seed, {})[method] = (batch.values.copy(), batch.observed.copy(), np.asarray(y).copy())
        return SimpleNamespace(predict=lambda x: np.zeros(len(x))), {'samples': len(y)}
    monkeypatch.setattr(fqi, 'fit_action', identical_inputs)
    regression_benchmark(tmp_path, 'synthetic', dataset_folder=problem)
    for methods in model_inputs.values():
        assert set(methods) == {'bart', 'cart', 'extra_trees'}
        for other in ('cart', 'extra_trees'):
            for left, right in zip(methods['bart'], methods[other]):
                np.testing.assert_array_equal(left, right)


def test_collection_resume_no_duplicate_transitions_or_administrative_terminal(tmp_path):
    from experiments.collect_pong_dataset import CollectionController, collected_arrays
    from experiments.pong_analysis_runtime import episode
    def factory():
        return CommonPongEnvironment(CommonPongConfig('visual', '0'*64, 20), driver=Driver())
    def run(folder, decision_limit=None):
        return episode(factory, CollectionController(assisted=True,
            directions={'up': 'RIGHT', 'down': 'LEFT'}, fraction=.5), folder,
            {'dataset': 'assisted', 'seed': 600201}, seed=600201, horizon=6, seconds=60,
            rng_seed=600701, administrative_horizon=True, decision_limit=decision_limit)
    whole = run(tmp_path/'whole')
    partial = run(tmp_path/'resumed', decision_limit=2)
    assert partial['status'] == 'partial' and partial['steps'] == 2
    resumed = run(tmp_path/'resumed')
    assert whole['transitions'] == resumed['transitions']
    data = collected_arrays([resumed])
    assert data['transition_ids'].tolist() == list(range(6))
    assert not data['terminated'].any() and not data['truncated'].any()
    assert data['next_present'].all()
    assert resumed['return_scope'] == 'administrative_quota'
    assert set(data['behavior_policies']) <= {'random', 'object_reactive'}
    assert data['history_reset'][0]
    assert data['state_values'].shape[1] == 20  # auxiliary columns do not enter the model


def test_portable_metadata_tampering_rejected(tmp_path):
    from experiments.freeze_pong_analysis_data import publish_portable
    from experiments.pong_analysis_common import portable_metadata
    from experiments.pong_analysis_runtime import bytes_hash
    raw = b'portable bytes'
    metadata = {'schema': 'pong_fixed_fit_v2', 'kind': 'fixed_fit', 'npz_sha256': bytes_hash(raw),
                'config': {'reward_min': -1., 'reward_max': 1.}}
    published = publish_portable(tmp_path, metadata, raw)
    assert portable_metadata(tmp_path) == published
    changed = {**published, 'config': {'reward_min': -.9, 'reward_max': 1.1}}
    (tmp_path/'metadata.json').write_text(json.dumps(changed))
    with pytest.raises(ValueError, match='metadata differs'):
        portable_metadata(tmp_path)
