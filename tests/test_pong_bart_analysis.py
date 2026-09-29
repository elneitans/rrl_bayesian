"""Small deterministic contracts; no reserved seeds or pilot checkpoints are loaded."""
from dataclasses import asdict
import pickle
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
import pytest

from bayesian_rrtl.encoding import RelationalEncoder, RelationKey
from bayesian_rrtl.objects import DetectedObject, ObjectFrame
from bayesian_rrtl.priors import FixedScale, NoisePrior, TreePrior
from bayesian_rrtl.q_config import BayesianQConfig
from bayesian_rrtl.replay import ReplayBuffer, Transition
from bayesian_rrtl.targets import frozen_q_targets
from experiments.audit_bart_fixed_data import run_chain, summarize
from experiments.compare_fqi_trees import (FittedQ, evaluate_model, fit_action, fit_epoch,
                                            nominal_indicators)
from experiments.evaluate_pong_heuristics import ReactivePolicy, probe_actions
from experiments.freeze_pong_analysis_data import fixed_transitions
from experiments.pong_analysis_common import config, episode_split, load_npz, read, transition_from_arrays
from preprocessing import Fact


def state(encoder, value=None):
    facts = [] if value is None else [Fact('logical', value, 'present', 'player_t')]
    return encoder.transform([facts])


def test_development_seed_and_episode_boundaries():
    path = Path(__file__).resolve().parents[1] / 'configs/pong_bart_analysis_v1.json'
    specification = config(path)
    assert not set(specification['development_seeds']) & set(range(420101, 420121))
    split, heldout = episode_split([1, 1, 2, 2, 3, 3, 4, 4], .25)
    assert heldout == [4]
    assert split.tolist() == [True]*6 + [False]*2
    value = read(path)
    value['action_probe_seed'] = value['development_seeds'][0]
    broken = path.parent / 'nonexistent.json'
    # Validate duplicate detection without changing the versioned config.
    import tempfile, json
    with tempfile.TemporaryDirectory() as folder:
        broken = Path(folder) / 'config.json'
        broken.write_text(json.dumps(value))
        with pytest.raises(ValueError, match='seeds'):
            config(broken)


def test_false_and_missing_are_distinct_for_extra_trees():
    encoder = RelationalEncoder([RelationKey('logical', 'present', 'player_t')])
    batch = encoder.transform([[], [Fact('logical', False, 'present', 'player_t')],
                               [Fact('logical', True, 'present', 'player_t')]])
    design = nominal_indicators(batch)
    assert design.tolist() == [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]
    config_data = asdict(BayesianQConfig(actions=('a',), gamma=.5,
        reward_min=0., reward_max=1., max_depth=2, min_child=1))
    values = encoder.transform([[], [],
        [Fact('logical', False, 'present', 'player_t')]]).take([0, 1, 2])
    fitted1, _ = fit_action('extra_trees', values, np.array([2., 2., 0.]), config_data, 88)
    fitted2, _ = fit_action('extra_trees', values, np.array([2., 2., 0.]), config_data, 88)
    np.testing.assert_array_equal(fitted1.predict(values), fitted2.predict(values))


def test_frozen_targets_terminal_and_truncation_and_tiny_fqi():
    encoder = RelationalEncoder([RelationKey('logical', 'present', 'player_t')])
    active, terminal = state(encoder, True), state(encoder, False)
    transitions = tuple(Transition(i, active, 0, 1., 1., active, False, i == 1, i, 0)
                        for i in range(2)) + tuple(
        Transition(i+2, terminal, 0, 0., 0., None, True, False, i+2, 0) for i in range(2))
    previous = SimpleNamespace(predict=lambda batch: np.full((len(batch), 1), 2.))
    np.testing.assert_array_equal(frozen_q_targets(transitions, previous, encoder, gamma=.5),
                                  [2., 2., 0., 0.])
    np.testing.assert_array_equal(frozen_q_targets(transitions, previous, encoder, gamma=.5,
                                                   bootstrap_on_truncation=False), [2., 1., 0., 0.])
    cfg = asdict(BayesianQConfig(actions=('a',), gamma=.5, reward_min=0., reward_max=1.,
                                max_depth=2, min_child=1))
    q0 = FittedQ(('a',), (None,), encoder.catalog_hash)
    q1, _ = fit_epoch('cart', transitions, encoder, cfg, q0, 1)
    q2, _ = fit_epoch('cart', transitions, encoder, cfg, q1, 2)
    assert q1.predict(active)[0, 0] == 1.
    assert q2.predict(active)[0, 0] == 1.5
    assert q2.predict(terminal)[0, 0] == 0.


def test_portable_transitions_keep_episode_ids_and_flags(tmp_path):
    encoder = RelationalEncoder([RelationKey('logical', 'present', 'player_t')])
    replay = ReplayBuffer(10)
    for i in range(4):
        replay.append(state=state(encoder, bool(i % 2)), action=0, reward=float(i),
            raw_reward=float(i), next_state=None if i == 3 else state(encoder, True),
            terminated=i == 3, truncated=i == 2, episode_id=i, step_id=0)
    agent = SimpleNamespace(replay=replay, encoder=encoder,
        config=BayesianQConfig(actions=('a',), gamma=.5, reward_min=0., reward_max=1.))
    meta = fixed_transitions(agent, None, tmp_path / 'frozen', batch_per_action=3,
                             fraction=.25, seed=15)
    data = load_npz(tmp_path / 'frozen/fixed_transitions.npz', meta['npz_sha256'])
    assert meta['holdout_episodes'] == [3]
    assert data['fqi_selected'].sum() == 3
    assert not data['fqi_selected'][3]
    assert transition_from_arrays(data, 2, encoder).truncated
    assert transition_from_arrays(data, 3, encoder).next_state is None
    assert transition_from_arrays(data, 3, encoder).terminated


def test_reactive_fact_policy_cannot_see_boxes_and_absence_fires():
    actions = ('NOOP', 'FIRE', 'RIGHT', 'LEFT')
    policy = ReactivePolicy('RIGHT', 'LEFT', actions)
    assert policy.fact_action([]) == 1
    assert policy.fact_action([]) == 0
    facts = [Fact('logical', True, 'present', 'player_t'),
             Fact('logical', True, 'present', 'ball_t'),
             Fact('comparative', 'less', 'y', 'player_t', 'ball_t')]
    assert policy.fact_action(facts) == 3
    reversed_facts = facts[:-1] + [Fact('comparative', 'more', 'y', 'player_t', 'ball_t')]
    assert policy.fact_action(reversed_facts) == 2
    assert 'environment' not in ReactivePolicy.fact_action.__code__.co_varnames


def test_action_probe_measures_movement_not_action_name():
    class Env:
        actions = ('NOOP', 'FIRE', 'RIGHT', 'LEFT')
        config = SimpleNamespace(extractor='ocatari_ram_v1')
        def __init__(self):
            self.y = 100.
            self.backend = SimpleNamespace(current_frame=None)
        def reset(self, *, seed):
            self.y = 100.
            self._frame()
        def _frame(self):
            self.backend.current_frame = ObjectFrame((
                DetectedObject('player', True, (0., self.y, 4., 8.)),
                DetectedObject('ball', True, (50., 100., 2., 2.))))
        def step(self, action):
            self.y += 2. if action == 2 else -2.
            self._frame()
            return [], 0., 0., False, False
        def close(self):
            pass
    probe = probe_actions(Env, 56)
    assert probe['up'] == 'LEFT' and probe['down'] == 'RIGHT'
    assert probe['displacements_y_pixels']['LEFT'] < 0


def test_evaluation_does_not_fit_or_mutate_frozen_model():
    encoder = RelationalEncoder([RelationKey('logical', 'present', 'player_t')])
    model = FittedQ(('NOOP',), (None,), encoder.catalog_hash)
    before = pickle.dumps(model)
    class Env:
        actions = ('NOOP',)
        def __init__(self): self.steps = 0
        def reset(self, *, seed): return []
        def encoder(self): return encoder
        def step(self, action):
            self.steps += 1
            return [], 0., -1., self.steps == 2, False
        def close(self): pass
    rows = evaluate_model(model, Env, (56, 57), 5, 10)
    assert len(rows) == 2 and all(r['raw_return'] == -2. for r in rows)
    assert all(r['return_scope'] == 'complete_game' for r in rows)
    assert pickle.dumps(model) == before


def test_fixed_audit_traces_include_burnin_and_stuck_is_not_pass(tmp_path):
    encoder = RelationalEncoder([RelationKey('logical', 'present', 'player_t')])
    batch = encoder.transform([[Fact('logical', bool(i % 2), 'present', 'player_t')] for i in range(4)])
    job = {'batch': batch, 'targets': np.array([0., 1., 0., 1.]),
           'reference': batch.take([0, 1]), 'reference_rule': ['a', 'b'],
           'prior': TreePrior(max_depth=2, min_child=1),
           'noise': NoisePrior(), 'scale': FixedScale(0., 1.), 'k': 2., 'm': 1,
           'data_hash': 'fixed', 'transition_ids_sha256': 'ids',
           'calibration_method': 'test', 'variance_floor_used': False}
    folder = tmp_path / 'audit'
    for chain in range(4):
        assert run_chain(job, folder / f'chain_{chain}', chain_seed=chain+200,
                         burn_in=2, draws=8, deadline=time.monotonic()+30,
                         checkpoint_every=2) == 'completed'
        with np.load(folder / f'chain_{chain}/trace.npz') as data:
            assert len(data['sigma2']) == 10 and data['predictions'].shape == (10, 2)
            assert len(data['grow_selected']) == 10
    summary = summarize(job, folder, 4, 2)
    assert summary['status'] == 'completed' and summary['data_hash'] == 'fixed'
    from bayesian_rrtl.diagnostics import chain_diagnostics
    assert chain_diagnostics(np.ones((4, 8)))['status'] == 'constant'
    assert not summary['all_observables_pass'] or all(
        v['status'] == 'ok' for v in summary['diagnostics'].values())
