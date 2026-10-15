"""B8–B12 acceptance: synthetic learning, exact policy, immutable resume and workers."""
import copy
import json
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import signal
import time

import numpy as np
import pytest

from bayesian_rrtl.comparison_checkpoint import (checkpoint_format, load_online_session, save_online_session)
from bayesian_rrtl.encoding import RelationKey, RelationalEncoder
from bayesian_rrtl.online.learners import (LearningConfig, OnlineBatchAgent, PredictionCache,
                                         nominal_indicators)
from bayesian_rrtl.online.protocol import authorize, dry_run, freeze, load, resolve
from bayesian_rrtl.online.session import OnlineSession
from bayesian_rrtl.online.storage import (binding, inventory, load_checkpoint, publish, stream_rows)
from experiments.compare_pong_progressive import (StopRequest, consolidate, evaluate_cell, train_cell)
from experiments.pong_analysis_runtime import checked_read, unit_lock
from preprocessing import Fact

ROOT = Path(__file__).resolve().parents[1]


class TinyEnvironment:
    """One-step nominal bandit: action 2 yields +1, others -1. No Pong performance evidence."""
    actions = ('NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE')

    def __init__(self, horizon=4, delay=0.):
        self.horizon, self.delay = horizon, delay
        self.steps = 0
        self.seed = None

    def encoder(self):
        return RelationalEncoder([RelationKey('logical', 'present', 'ball')])

    def reset(self, *, seed):
        self.seed = seed
        self.steps = 0
        return [Fact('logical', 'False', 'present', 'ball')]

    def step(self, action):
        if self.delay:
            time.sleep(self.delay)
        self.steps += 1
        raw = 1. if action == 2 else -1.
        return [], raw, raw, True, False

    def snapshot(self):
        return {'steps': self.steps, 'seed': self.seed}

    def restore(self, saved):
        self.steps, self.seed = saved['steps'], saved['seed']

    def runtime_manifest(self):
        return {'kind': 'simulated_nominal_bandit_v1', 'horizon': self.horizon}

    def close(self):
        pass


class TinyFactory:
    def __init__(self, horizon=4, delay=0.):
        self.horizon, self.delay = horizon, delay

    def __call__(self):
        return TinyEnvironment(self.horizon, self.delay)

    def with_horizon(self, horizon):
        return TinyFactory(horizon, self.delay)


def protocol(learner=None):
    p = json.loads((ROOT/'configs/pong_online_smoke.json').read_text())
    if learner:
        p['level'] = 'baseline'
        p['learners'] = {learner: p['learners'].get(learner, {'n_estimators': 3, 'max_depth': 2, 'min_samples_leaf': 1})}
    return p


def campaign(folder, learner='cart'):
    p = protocol(learner)
    if learner == 'extra_trees':
        from bayesian_rrtl.online.learners import sklearn_version
        installed = sklearn_version()
        if installed is None:
            pytest.skip('Optional ExtraTrees dependency not installed')
        # Tests declare their actual version explicitly, as a separate scientific config.
        # Production defaults and Linux dependency pins remain unchanged.
        p['learners']['extra_trees']['sklearn_version'] = installed
    p['pipelines'] = ['visual']
    p['max_decisions'], p['milestones'] = 64, [32, 64]
    freeze(p, folder)
    authorize(folder, 64, 'Discardable synthetic recovery acceptance')
    return load(folder)['cells'][0]


def payload(folder, cell):
    status = checked_read(folder/cell['id']/'status.json')
    saved, _ = load_checkpoint(folder/cell['id'], status['checkpoint'], binding(load(folder), cell))
    return saved, list(stream_rows(folder/cell['id'], status['log_head']))


def worker(folder, learner, limit=None, delay=0.):
    train_cell(folder, f'visual/{learner}/501501', factory=TinyFactory(delay=delay), decision_limit=limit)


def killed_worker(folder):
    def kill_after_checkpoint(counter):
        if counter == 16:
            os.kill(os.getpid(), signal.SIGKILL)
    train_cell(folder, 0, factory=TinyFactory(), after_publish=kill_after_checkpoint)


def test_sigkill_resume_measurements_preserve_unknown_attempt_costs(tmp_path):
    from experiments.compare_pong_progressive import measurements
    folder = tmp_path/'killed-costs'
    cell = campaign(folder)
    context = mp.get_context('spawn')
    killed = context.Process(target=killed_worker, args=(folder,))
    killed.start()
    try:
        killed.join(40)
        assert not killed.is_alive() and killed.exitcode == -signal.SIGKILL
    finally:
        if killed.is_alive():
            killed.kill()
            killed.join(5)
    path = folder/cell['id']
    assert checked_read(path/'status.json')['counter'] == 16
    attempt_path, = (path/'attempts').glob('*.json')
    incomplete_bytes = attempt_path.read_bytes()
    dead = checked_read(attempt_path)
    assert dead['ended_unix'] is dead['seconds'] is dead['peak_rss_bytes'] is None
    resumed = context.Process(target=worker, args=(folder, 'cart'))
    resumed.start()
    try:
        resumed.join(40)
        assert not resumed.is_alive() and resumed.exitcode == 0
    finally:
        if resumed.is_alive():
            resumed.kill()
            resumed.join(5)
    assert attempt_path.read_bytes() == incomplete_bytes
    saved, rows = payload(folder, cell)
    assert saved['counter'] == len(rows) == 64
    assert len({r['transition_id'] for r in rows}) == 64
    report = consolidate(folder)['cells'][0]
    assert report['training_state'] == 'completed'
    assert report['unfinished_training_attempts'] == 1
    assert report['peak_rss_bytes'] is None and report['peak_rss_bytes_known_max'] > 0
    row, = measurements(folder)
    assert row['decisions'] == 64 and row['unfinished_training_attempts'] == 1
    assert row['training_seconds'] is None and row['training_seconds_recorded_lower_bound'] > 0
    assert row['peak_rss_bytes'] is None and row['peak_rss_bytes_known_max'] > 0
    assert len(row['training_attempts']) == 2
    incomplete, = [a for a in row['training_attempts'] if a['record_status'] == 'incomplete']
    assert incomplete['ended_unix'] is incomplete['seconds'] is incomplete['peak_rss_bytes'] is None
    # Export remains strict JSON; unknown metrics never become zero/NaN.
    json.dumps(row, allow_nan=False)
    # Older interrupted attempt records omitted these fields altogether.
    from experiments.pong_analysis_runtime import checked_write
    checked_write(attempt_path, {k:v for k,v in dead.items()
                               if k not in ('ended_unix', 'seconds', 'peak_rss_bytes', 'state')})
    legacy, = measurements(folder)
    assert legacy['training_seconds'] is None and legacy['peak_rss_bytes'] is None
    assert legacy['unfinished_training_attempts'] == 1


def test_protocol_single_learner_and_pending_long():
    p = json.loads((ROOT/'configs/pong_online_baseline.json').read_text())
    r = resolve(p)
    assert len(r['cells']) == 6 and all(c['learner'] == 'corrected_common' for c in r['cells'])
    assert r['protocol']['milestones'] == [100000, 500000, 2000000]
    long = json.loads((ROOT/'configs/pong_online_long_pending.json').read_text())
    d = dry_run(long)
    assert len(d['workload']) == 30 and d['estimates'] == [] and d['pending_authorizations']
    assert 'max_decisions' in d['resolved']['unresolved'] and 'learners.bart' in d['resolved']['unresolved']
    assert all(c['decisions'] is None for c in d['workload'])
    bad = copy.deepcopy(p)
    bad['evaluation']['final_development']['seeds'][0] = bad['evaluation']['reserved_test']['seeds'][0]
    with pytest.raises(ValueError, match='disjoint'):
        resolve(bad)
    bad = copy.deepcopy(p)
    bad['learners']['unknown'] = {}
    with pytest.raises(ValueError, match='learner'):
        resolve(bad)


@pytest.mark.parametrize('learner', ['corrected_common', 'cart', 'bart', 'extra_trees'])
def test_same_runner_learns_known_terminal_mdp(tmp_path, learner):
    cell = campaign(tmp_path/learner, learner)
    status = train_cell(tmp_path/learner, 0, factory=TinyFactory())
    assert status['state'] == 'completed' and status['counter'] == 64
    saved, rows = payload(tmp_path/learner, cell)
    agent = saved['session']['collector']['agent']
    state = TinyEnvironment().reset(seed=0)
    q = agent.predict_q(state) if learner == 'corrected_common' else agent.model.predict(agent.encode_state(state))[0]
    assert np.argmax(q) == 2
    if learner in ('cart', 'extra_trees'):
        assert q[2] == 1 and all(q[i] == -1 for i in range(6) if i != 2)
        assert all(v is None for v in agent.calibrations)
    assert all(r['terminated'] and not r['truncated'] for r in rows)
    assert 'points_won' in rows[-1]
    evaluations = evaluate_cell(tmp_path/learner, 0, factory=TinyFactory())
    assert all(e['state'] == 'completed' for e in evaluations)
    assert consolidate(tmp_path/learner)['completed']
    assert payload(tmp_path/learner, cell)[1] == rows


def fixed_agents():
    learning = LearningConfig(('a', 'b'), seed=17, gamma=.5, replay_capacity=16,
        batch_size_per_action=8, fit_interval=4, warmup_interactions=0, epsilon_decay_steps=50000,
        validation_seeds=(777,), test_seeds=(888,))
    enc = TinyEnvironment().encoder()
    agents = [OnlineBatchAgent(learning, enc, learner=method,
              estimator={'max_depth': 2, 'min_child': 1} if method == 'cart' else
              {'m': 1, 'burn_in': 1, 'draws': 2, 'max_depth': 2}) for method in ('cart', 'bart')]
    class Reference:
        model_id = 12
        posteriors = (None, None)
        def predict(self, batch):
            return np.tile([4., 6.], (len(batch), 1))
    for agent in agents:
        state = [Fact('logical', 'False', 'present', 'ball')]
        for i in range(4):
            agent.observe(state, i%2, 1., state, i == 0, i == 1,
                          raw_reward=1., episode_id=0, step_id=i)
        agent.model = Reference()
    return agents


def test_fixed_targets_frozen_whole_block_and_failed_publication():
    cart, bart = fixed_agents()
    cart.fit_if_due()
    bart.fit_if_due()
    for left, right in zip(cart.last_fit.datasets, bart.last_fit.datasets):
        assert left.transition_ids == right.transition_ids
        np.testing.assert_array_equal(left.batch.values, right.batch.values)
        np.testing.assert_array_equal(left.targets, right.targets)
        assert set(left.targets) <= {1., 4.}  # terminal reward, truncation/nonterminal bootstrap
    cart, _ = fixed_agents()
    old = cart.model
    before = pickle.dumps((cart.replay_rng, cart.mcmc_rngs, cart.calibrations))
    original = cart.fit_action
    def fail(data, rng, calibration):
        if data.action == 1:
            raise RuntimeError('second action failed')
        return original(data, rng, calibration)
    cart.fit_action = fail
    with pytest.raises(RuntimeError):
        cart.fit_if_due()
    assert cart.model is old and cart.last_fit is None
    assert pickle.dumps((cart.replay_rng, cart.mcmc_rngs, cart.calibrations)) == before


@pytest.mark.parametrize('learner', ['corrected_common', 'cart', 'bart', 'extra_trees'])
def test_exact_resume_checkpoint_type_and_policy(tmp_path, learner):
    full, resumed = tmp_path/'full', tmp_path/'resumed'
    cell = campaign(full, learner)
    campaign(resumed, learner)
    train_cell(full, 0, factory=TinyFactory())
    train_cell(resumed, 0, factory=TinyFactory(), decision_limit=19)
    assert checked_read(resumed/cell['id']/'status.json')['counter'] == 19
    train_cell(resumed, 0, factory=TinyFactory())
    a, x = payload(full, cell)
    b, y = payload(resumed, cell)
    # Timing is operational; all decision/reward/target identities must agree exactly.
    def decisions(rows):
        return [{k:v for k,v in r.items() if k != 'fit'} for r in rows]
    assert decisions(x) == decisions(y)
    aa, bb = a['session']['collector']['agent'], b['session']['collector']['agent']
    assert type(aa) is type(bb)
    state = TinyEnvironment().reset(seed=0)
    if learner != 'corrected_common':
        np.testing.assert_array_equal(aa.model.predict(aa.encode_state(state)), bb.model.predict(bb.encode_state(state)))
        assert aa.replay.next_id == bb.replay.next_id == 64
        assert aa.replay_rng.bit_generator.state == bb.replay_rng.bit_generator.state
        for one, two in zip(aa.mcmc_rngs, bb.mcmc_rngs):
            assert one.bit_generator.state == two.bit_generator.state
    else:
        np.testing.assert_array_equal(aa.predict_q(state), bb.predict_q(state))
    assert a['session']['collector']['exploration_rng'].bit_generator.state == b['session']['collector']['exploration_rng'].bit_generator.state
    assert y[19]['epsilon'] < y[0]['epsilon']
    times = len(list((resumed/cell['id']/'attempts').glob('*.json')))
    train_cell(resumed, 0, factory=TinyFactory())
    assert len(list((resumed/cell['id']/'attempts').glob('*.json'))) == times
    assert checkpoint_format(resumed/cell['id']/checked_read(resumed/cell['id']/'status.json')['checkpoint']['path']) == 'pong_progressive_checkpoint_v2'


def test_reference_cache_false_missing_and_memory_bound():
    enc = TinyEnvironment().encoder()
    batch = enc.transform([[Fact('logical', 'False', 'present', 'ball')], []])
    indicators = nominal_indicators(batch)
    np.testing.assert_array_equal(indicators, [[0, 1, 0], [1, 0, 0]])
    agent, _ = fixed_agents()
    cache = PredictionCache(1300)
    before = pickle.dumps(agent.exploration_rng)
    expected = agent.model.predict(batch)
    np.testing.assert_array_equal(cache.predict(agent.model, batch), expected)
    np.testing.assert_array_equal(cache.predict(agent.model, batch), expected)
    assert cache.hits == 1 and cache.bytes <= 1300
    agent.model.model_id += 1
    cache.predict(agent.model, batch)
    assert cache.misses == 2
    assert pickle.dumps(agent.exploration_rng) == before


def test_cache_on_off_exact_decisions_and_learning():
    cell = resolve(protocol('cart'))['cells'][0]
    one = OnlineSession(cell, TinyEnvironment(), cache_bytes=0)
    two = OnlineSession(cell, TinyEnvironment(), cache_bytes=1048576)
    for _ in range(64):
        left, right = one.step(), two.step()
        assert {k:v for k,v in left.items() if k != 'fit'} == {k:v for k,v in right.items() if k != 'fit'}
        np.testing.assert_array_equal(one.agent.model.predict(one.agent.encode_state(one.state)),
                                      two.agent.model.predict(two.agent.encode_state(two.state)))
    assert one.exploration_rng.bit_generator.state == two.exploration_rng.bit_generator.state
    assert one.agent.replay_rng.bit_generator.state == two.agent.replay_rng.bit_generator.state
    assert two.agent.cache.bytes <= two.agent.cache.max_bytes


def test_unknown_index_code_changes_and_evaluation_pending(tmp_path, monkeypatch):
    cell = campaign(tmp_path/'c')
    with pytest.raises(ValueError, match='index'):
        train_cell(tmp_path/'c', 999, factory=TinyFactory())
    assert not (tmp_path/'c'/'visual').exists()
    train_cell(tmp_path/'c', 0, factory=TinyFactory())
    report = consolidate(tmp_path/'c')
    assert not report['completed'] and report['cells'][0]['comparison_state'] == 'pending_evaluation'
    assert report['reserved_test_executed'] is False
    with pytest.raises(ValueError, match='Reserved'):
        evaluate_cell(tmp_path/'c', 0, group='reserved_test', factory=TinyFactory())
    import bayesian_rrtl.online.protocol as module
    monkeypatch.setattr(module, 'source_hashes', lambda: {})
    with pytest.raises(ValueError, match='Code'):
        train_cell(tmp_path/'c', 0, factory=TinyFactory())
    assert cell['learner'] == 'cart'


def test_interrupted_segment_publication_preserves_frontier(tmp_path, monkeypatch):
    folder = tmp_path/'c'
    cell = campaign(folder)
    train_cell(folder, 0, factory=TinyFactory(), decision_limit=16)
    before = checked_read(folder/cell['id']/'status.json')
    import bayesian_rrtl.online.storage as store
    real = store.checked_write
    def fail(path, value, **kwargs):
        if Path(path).name == 'status.json':
            raise OSError('simulated atomic frontier failure')
        return real(path, value, **kwargs)
    monkeypatch.setattr(store, 'checked_write', fail)
    m = load(folder)
    data, _ = load_checkpoint(folder/cell['id'], before['checkpoint'], binding(m, cell))
    session = OnlineSession.restore(data['session'], cell, TinyEnvironment())
    session.step()
    with pytest.raises(OSError):
        publish(folder/cell['id'], m, cell, session, before, state='partial')
    assert checked_read(folder/cell['id']/'status.json') == before
    assert inventory(folder/cell['id'], before)['unconfirmed_logs']
    monkeypatch.setattr(store, 'checked_write', real)
    train_cell(folder, 0, factory=TinyFactory())
    saved, rows = payload(folder, cell)
    assert len(rows) == saved['counter'] == 64 and [r['transition_id'] for r in rows] == list(range(64))


def test_separate_process_recovery_and_lock(tmp_path):
    folder = tmp_path/'c'
    cell = campaign(folder)
    ctx = mp.get_context('spawn')
    p = ctx.Process(target=worker, args=(folder, 'cart', 19))
    p.start()
    p.join(30)
    assert p.exitcode == 0
    q = ctx.Process(target=worker, args=(folder, 'cart'))
    q.start()
    q.join(30)
    assert q.exitcode == 0
    assert payload(folder, cell)[0]['counter'] == 64
    with unit_lock(folder/cell['id']):
        with pytest.raises(RuntimeError, match='Concurrent'):
            train_cell(folder, 0, factory=TinyFactory())
    first = evaluate_cell(folder, 0, factory=TinyFactory())
    identities = {p: p.read_bytes() for p in (folder/cell['id']/'evaluations').rglob('state.json')}
    assert evaluate_cell(folder, 0, factory=TinyFactory()) == first
    assert all(p.read_bytes() == raw for p, raw in identities.items())
    assert consolidate(folder)['completed']


def test_standalone_cart_session_and_historic_recognition(tmp_path):
    cell = resolve(protocol('cart'))['cells'][0]
    s = OnlineSession(cell, TinyEnvironment())
    for _ in range(16):
        s.step()
    path = save_online_session(s, tmp_path/'cart.pkl')
    restored = load_online_session(path, cell=cell, environment=TinyEnvironment())
    assert restored.learner == 'cart' and restored.agent.calibrations == (None,)*6
    assert checkpoint_format(path) == 'pong_online_session_v2'
    from bayesian_rrtl.checkpoint import MAGIC
    (tmp_path/'old.rrtl').write_bytes(MAGIC+b'placeholder')
    assert checkpoint_format(tmp_path/'old.rrtl') == 'bart_historic_v1'
    from bayesian_rrtl.baseline_checkpoint import MAGIC
    (tmp_path/'old.rrtl').write_bytes(MAGIC+b'placeholder')
    assert checkpoint_format(tmp_path/'old.rrtl') == 'rrtl_common_historic_v1'


def test_signal_handler_only_requests_stop():
    s = StopRequest()
    s.handler(15, None)
    assert s.requested and s.signal == 15


def test_budget_extension_and_authorized_milestone_preserve_counters(tmp_path):
    folder = tmp_path/'budget'
    p = protocol('cart')
    p['pipelines'] = ['visual']
    p['budgets']['training_seconds_per_cell'] = .00001
    freeze(p, folder)
    authorize(folder, 32, 'First discardable milestone')
    first = train_cell(folder, 0, factory=TinyFactory())
    assert first['counter'] == 0 and first['reason'] == 'training_budget_exhausted'
    second = train_cell(folder, 0, factory=TinyFactory(), extend_seconds=60,
                        extension_id='review-one', extension_reason='Discardable local verification')
    assert second['counter'] == 32 and second['reason'] == 'authorized_milestone_reached'
    authorize(folder, 64, 'Second discardable milestone')
    last = train_cell(folder, 0, factory=TinyFactory(), extend_seconds=60,
                      extension_id='review-one', extension_reason='Discardable local verification')
    assert last['counter'] == 64 and last['state'] == 'completed'
    budget = checked_read(folder/load(folder)['cells'][0]['id']/'budget.json')
    assert len(budget['extensions']) == 1 and budget['elapsed_seconds'] > 0
    changed = copy.deepcopy(p)
    changed['exploration']['epsilon_decay_steps'] *= 2
    with pytest.raises(ValueError, match='variant'):
        freeze(changed, folder)


def test_failure_state_rolls_back_then_resume_has_no_duplicates(tmp_path):
    folder = tmp_path/'failure'
    cell = campaign(folder)
    class Broken(TinyEnvironment):
        def step(self, action):
            raise RuntimeError('simulated update failure')
    with pytest.raises(RuntimeError, match='update failure'):
        train_cell(folder, 0, factory=lambda: Broken())
    assert consolidate(folder)['cells'][0]['comparison_state'] == 'failed'
    train_cell(folder, 0, factory=TinyFactory())
    _, rows = payload(folder, cell)
    assert [r['transition_id'] for r in rows] == list(range(64))


def test_extra_trees_incompatible_dependency_and_encoding():
    c = LearningConfig(('a', 'b'), seed=1)
    with pytest.raises(ValueError, match='dependency'):
        OnlineBatchAgent(c, TinyEnvironment().encoder(), learner='extra_trees',
                         estimator={'sklearn_version': '0.0-impossible'})
    with pytest.raises(ValueError, match='encoding'):
        OnlineBatchAgent(c, TinyEnvironment().encoder(), learner='extra_trees',
                         estimator={'encoding': 'ordinal'})


def test_real_signal_recovery_command_simulated(tmp_path):
    from experiments.validate_pong_online_recovery import validate
    report = validate(tmp_path/'recovery', selection=1)  # CART, separate worker processes and actual USR1
    row = report['cells'][0]
    assert row['signal_recovery'] and row['exact_decisions_models_rng'] and row['second_writer_rejected']
    assert row['completed_evaluation_reused'] and row['no_duplicate_transitions']
    assert report['reserved_test_executed'] is False


def test_same_counter_status_preserves_evaluation_checkpoint(tmp_path):
    folder = tmp_path/'stable'
    cell = campaign(folder)
    train_cell(folder, 0, factory=TinyFactory(), decision_limit=32)
    status = checked_read(folder/cell['id']/'status.json')
    ref = status['checkpoint']
    evaluate_cell(folder, 0, point=32, group='intermediate', factory=TinyFactory())
    train_cell(folder, 0, factory=TinyFactory(), decision_limit=0)
    assert checked_read(folder/cell['id']/'status.json')['checkpoint'] == ref
    train_cell(folder, 0, factory=TinyFactory())
    # Earlier evaluation remains bound to the same bytes after final publication.
    evaluate_cell(folder, 0, point=32, group='intermediate', factory=TinyFactory())


def test_retention_preserves_calendar_backup_and_bounded_segments(tmp_path):
    from bayesian_rrtl.online.storage import cleanup
    folder = tmp_path/'retention'
    cell = campaign(folder)
    train_cell(folder, 0, factory=TinyFactory())
    path = folder/cell['id']
    status = checked_read(path/'status.json')
    plan = inventory(path, status, [0, 32, 64])
    assert len(plan['artifacts']) == 5  # initial plus four intervals; no duplicate final checkpoint
    assert all(a['bytes'] > 0 for a in plan['artifacts'])
    before = list(stream_rows(path, status['log_head']))
    cleanup(path, status, [0, 32, 64])
    assert list(stream_rows(path, status['log_head'])) == before
    evaluate_cell(folder, 0, factory=TinyFactory())
    assert consolidate(folder)['completed']


def test_simulated_measurements_cannot_estimate_atari(tmp_path):
    from experiments.compare_pong_progressive import measurements
    folder = tmp_path/'measure'
    campaign(folder)
    train_cell(folder, 0, factory=TinyFactory())
    observed = measurements(folder)
    assert observed[0]['runtime_environment']['kind'].startswith('simulated')
    assert dry_run(protocol('cart'), observed)['estimates'] == []


def test_partial_evaluation_failure_reuses_completed_episodes(tmp_path):
    folder = tmp_path/'eval-recovery'
    cell = campaign(folder)
    train_cell(folder, 0, factory=TinyFactory())
    class Broken(TinyEnvironment):
        def step(self, action):
            raise RuntimeError('second evaluation failed')
    class FailsSecond(TinyFactory):
        def __init__(self, horizon=4):
            super().__init__(horizon)
            self.calls = 0
        def __call__(self):
            self.calls += 1
            return Broken(self.horizon) if self.calls == 2 else TinyEnvironment(self.horizon)
        def with_horizon(self, horizon):
            return self
    with pytest.raises(RuntimeError, match='second evaluation'):
        evaluate_cell(folder, 0, factory=FailsSecond())
    path = folder/cell['id']/'evaluations'/'intermediate'/'0'/'501801'/'state.json'
    completed = path.read_bytes()
    report = consolidate(folder)
    assert report['cells'][0]['comparison_state'] == 'failed'
    assert any(r['reason'] == 'evaluation_failed' for r in report['cells'][0]['pending_evaluations'])
    evaluate_cell(folder, 0, factory=TinyFactory())
    assert path.read_bytes() == completed and consolidate(folder)['completed']


def test_full_smoke_freezes_with_canonical_learner_order(tmp_path):
    p = protocol()
    m = freeze(p, tmp_path/'smoke')
    assert load(tmp_path/'smoke') == m and len(m['cells']) == 6
    assert [c['learner'] for c in m['cells'][:3]] == ['corrected_common', 'cart', 'bart']



def test_json_measurements_match_compatible_common_protocol():
    from bayesian_rrtl.games.pong import PongRelationsV1
    from bayesian_rrtl.online.protocol import runtime, source_hashes
    from experiments.pong_common_contract import CommonPongConfig
    p = protocol('cart')
    c = resolve(p)['cells'][0]
    row = {'environment': c['environment'], 'learner': c['learner'],
           'runtime_environment': {'driver_provenance': 'verified_atari',
                'contract': CommonPongConfig(**c['environment']).manifest()},
           'catalog_hash': PongRelationsV1().encoder().catalog_hash,
           'cache_bytes': p['cache_bytes'],
           'evaluation_horizons': {k:g['horizon'] for k,g in p['evaluation'].items() if k != 'reserved_test'},
           'learning_without_seed': {k:v for k,v in c['learning'].items() if k != 'seed'},
           'estimator': c['estimator'], 'runtime': runtime('cart'), 'source_hashes': source_hashes(),
           'training_seconds': 2., 'decisions': 64, 'log_bytes': 1000,
           'checkpoint_blob_bytes_range': [100, 200], 'peak_rss_bytes': 10000,
           'evaluation_episode_seconds': [1., 2.], 'evaluation_worker_seconds_per_completed_episode': 2.5}
    serialized = json.loads(json.dumps([row]))
    # JSON changes tuple actions/seeds into lists. This must not reject compatible cost evidence.
    estimates = dry_run(p, serialized)['estimates']
    assert len(estimates) == 1  # one smoke seed; never reuse across different pipelines
    assert estimates[0]['training_seconds_observed_rate_range'] == [2., 2.]
    unknown = {**serialized[0], 'training_seconds': None, 'peak_rss_bytes': None}
    assert dry_run(p, [unknown])['estimates'] == []
    mixed = dry_run(p, [unknown, *serialized])['estimates'][0]
    assert mixed['training_seconds_observed_rate_range'] == [2., 2.]
    assert mixed['peak_rss_observed_range'] == [10000, 10000]
    assert mixed['incomplete_training_cost_records'] == mixed['unknown_peak_rss_records'] == 1
    no_memory = dry_run(p, [{**serialized[0], 'peak_rss_bytes': None}])['estimates'][0]
    assert no_memory['peak_rss_observed_range'] is None
    serialized[0]['learning_without_seed']['epsilon_decay_steps'] += 1
    assert dry_run(p, serialized)['estimates'] == []


def test_attempt_limit_preserves_frozen_budget_and_safe_frontier(tmp_path):
    folder = tmp_path/'attempt-limit'
    cell = campaign(folder)
    status = train_cell(folder, 0, factory=TinyFactory(), attempt_seconds=.000001)
    assert status['state'] == 'partial' and status['reason'] == 'attempt_time_exhausted'
    assert status['counter'] == 0
    train_cell(folder, 0, factory=TinyFactory(), attempt_seconds=60.)
    saved, rows = payload(folder, cell)
    assert saved['counter'] == len(rows) == 64
    assert checked_read(folder/cell['id']/'budget.json')['extensions'] == []


@pytest.mark.parametrize('options', [
    {'extend_seconds': -1., 'extension_id': 'bad', 'extension_reason': 'bad'},
    {'extend_seconds': float('nan'), 'extension_id': 'bad', 'extension_reason': 'bad'},
    {'extend_seconds': 1., 'extension_reason': 'missing ID'},
    {'extension_id': 'missing-amount'},
    {'extend_seconds': 1., 'extension_id': 'bad', 'extension_reason': '   '},
])
def test_invalid_extension_creates_no_cell_artifacts(tmp_path, options):
    folder = tmp_path/'invalid-extension'
    cell = campaign(folder)
    with pytest.raises(ValueError):
        train_cell(folder, 0, factory=TinyFactory(), **options)
    assert not (folder/cell['id']).exists()



def test_report_handles_active_evaluation_without_reading_transient_blobs(tmp_path):
    folder = tmp_path/'eval-active'
    cell = campaign(folder)
    train_cell(folder, 0, factory=TinyFactory())
    evaluate_cell(folder, 0, factory=TinyFactory())
    unit = folder/cell['id']/'evaluations'/'intermediate'/'32'/'501801'
    with unit_lock(unit):
        report = consolidate(folder)
    assert not report['completed']
    assert any(p['reason'] == 'evaluation_running' for p in report['cells'][0]['pending_evaluations'])
    assert consolidate(folder)['completed']
