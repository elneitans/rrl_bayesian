"""A1/A2 acceptance: no emulator, historical checkpoints or reserved seeds."""
import json
import multiprocessing
from pathlib import Path

import pytest

from bayesian_rrtl.encoding import RelationalEncoder, RelationKey
from bayesian_rrtl.games.pong import PongRelationsV1
from bayesian_rrtl.objects import DetectedObject, ObjectFrame
from experiments.evaluate_pong_heuristics import FactAdapter, ReactivePolicy, assisted_gate
from experiments.pong_analysis_runtime import (Budget, WorkUnit, episode, unit_lock)
from preprocessing import Fact


class ToyEnvironment:
    actions = ('NOOP', 'FIRE', 'RIGHT', 'LEFT')
    resets = 0
    decisions = []
    def __init__(self):
        self.steps = 0
    def reset(self, *, seed):
        type(self).resets += 1
        self.steps = 0
        return []
    def encoder(self):
        return RelationalEncoder([RelationKey('logical', 'present', 'player_t')])
    def step(self, action):
        type(self).decisions.append(action)
        self.steps += 1
        return [], 0., float(action == 1)-float(action == 2), self.steps == 7, False
    def snapshot(self):
        return {'steps': self.steps}
    def restore(self, saved):
        self.steps = saved['steps']
    def close(self):
        pass


class ToyController:
    def __init__(self):
        self.memory = 0
        self.log = []
    def action(self, state, env, rng):
        self.memory += 1
        action = (int(rng.integers(4))+self.memory) % 4
        self.log.append(action)
        return action


def run(folder, **options):
    return episode(ToyEnvironment, ToyController(), folder,
                   {'model': 'a', 'dataset': 'a', 'horizon': 20, 'mode': 'full'},
                   seed=600001, rng_seed=600101, horizon=20, seconds=60, **options)


def without_timing(row):
    return {k: v for k, v in row.items() if k != 'seconds'}


def test_exact_episode_resume_and_no_repeated_completed_work(tmp_path):
    whole = run(tmp_path/'whole')
    ToyEnvironment.decisions = []
    partial = run(tmp_path/'resume', decision_limit=3)
    assert partial['status'] == 'partial' and partial['steps'] == 3
    assert not partial['terminated'] and not partial['truncated']
    resumed = run(tmp_path/'resume')
    assert without_timing(whole) == without_timing(resumed)
    assert ToyEnvironment.decisions == whole['decision_log']
    resets = ToyEnvironment.resets
    assert run(tmp_path/'resume') == resumed
    assert ToyEnvironment.resets == resets


class FailingEnvironment(ToyEnvironment):
    fail_once = True
    def step(self, action):
        result = super().step(action)
        if self.steps == 5 and type(self).fail_once:
            type(self).fail_once = False
            raise RuntimeError('decision interrupted after emulator advanced')
        return result


def test_failed_decision_restores_last_frontier_instead_of_stitching(tmp_path):
    expected = {'model': 'fixed', 'horizon': 20}
    options = {'seed': 600001, 'rng_seed': 600101, 'horizon': 20, 'seconds': 60, 'checkpoint_every': 2}
    whole = episode(ToyEnvironment, ToyController(), tmp_path/'whole', expected, **options)
    FailingEnvironment.fail_once = True
    with pytest.raises(RuntimeError, match='interrupted'):
        episode(FailingEnvironment, ToyController(), tmp_path/'failed', expected, **options)
    resumed = episode(FailingEnvironment, ToyController(), tmp_path/'failed', expected, **options)
    assert without_timing(resumed) == without_timing(whole)


@pytest.mark.parametrize('field', ['model', 'dataset', 'horizon', 'mode'])
def test_changed_identity_rejected(tmp_path, field):
    original = {'model': 'a', 'dataset': 'a', 'horizon': 20, 'mode': 'full'}
    with unit_lock(tmp_path):
        WorkUnit(tmp_path, original).commit('completed', result={'answer': 1})
        with pytest.raises(ValueError, match='identity mismatch'):
            WorkUnit(tmp_path, original | {field: 'different'})


def test_corrupt_json_and_interrupted_commit_are_not_results(tmp_path, monkeypatch):
    import experiments.pong_analysis_runtime as runtime
    with unit_lock(tmp_path):
        unit = WorkUnit(tmp_path, {'identity': 1})
        original = unit.commit('partial', result={'steps': 1}, artifacts={'progress': b'old'})
        replace = runtime.os.replace
        def failing_replace(source, target):
            if Path(target).name == 'state.json':
                raise OSError('simulated atomic publication crash')
            replace(source, target)
        monkeypatch.setattr(runtime.os, 'replace', failing_replace)
        with pytest.raises(OSError):
            unit.commit('completed', result={'steps': 2}, artifacts={'progress': b'new'})
        assert unit.load() == original
        monkeypatch.setattr(runtime.os, 'replace', replace)
        (tmp_path/'state.json').write_text('{truncated')
        with pytest.raises(ValueError, match='Corrupt'):
            unit.load()


def test_corrupt_binary_rejected_before_loading(tmp_path):
    with unit_lock(tmp_path):
        unit = WorkUnit(tmp_path, {'identity': 1})
        saved = unit.commit('partial', artifacts={'progress': b'pickle-data'})
        ref = saved['artifacts']['progress']
        (tmp_path/'objects'/ref['path']).write_bytes(b'changed')
        with pytest.raises(ValueError, match='checksum'):
            unit.load()


def test_identity_reuse_canonicalizes_json_container_types(tmp_path):
    with unit_lock(tmp_path):
        WorkUnit(tmp_path, {'actions': ('NOOP', 'FIRE')}).commit('completed', result={1: 'ok'})
        unit = WorkUnit(tmp_path, {'actions': ('NOOP', 'FIRE')})
        assert unit.load()['result'] == {'1': 'ok'}


def test_historical_analysis_is_read_only_without_migration(tmp_path):
    from experiments.pong_analysis_common import load_manifest
    path = tmp_path/'manifest.json'
    content = json.dumps({'schema': 'pong_bart_analysis_v1', 'source_hashes': {}})
    path.write_text(content)
    assert load_manifest(tmp_path)['schema'] == 'pong_bart_analysis_v1'
    with pytest.raises(ValueError, match='read-only'):
        load_manifest(tmp_path, writable=True)
    assert path.read_text() == content


def lock_contender(folder, connection):
    try:
        with unit_lock(folder):
            connection.send('unexpected_success')
    except RuntimeError:
        connection.send('conflict')
    finally:
        connection.close()


def test_multiprocess_single_writer(tmp_path):
    ctx = multiprocessing.get_context('spawn')
    first, second = ctx.Pipe()
    with unit_lock(tmp_path):
        process = ctx.Process(target=lock_contender, args=(str(tmp_path), second))
        process.start()
        assert first.poll(20) and first.recv() == 'conflict'
        process.join(20)
        assert process.exitcode == 0
    with unit_lock(tmp_path):
        WorkUnit(tmp_path, {'ok': True}).commit('completed')


def test_budget_extension_is_explicit_and_idempotent(tmp_path):
    with unit_lock(tmp_path):
        budget = Budget(tmp_path, 1)
        budget.value['elapsed_seconds'] = 2
        from experiments.pong_analysis_runtime import checked_write
        checked_write(budget.path, budget.value)
        assert Budget(tmp_path, 1).remaining == 0
        extended = Budget(tmp_path, 1, extend_seconds=4, extension_id='one')
        assert 2.9 < extended.remaining <= 3
        assert len(Budget(tmp_path, 1, extend_seconds=4, extension_id='one').value['extensions']) == 1
        with pytest.raises(ValueError, match='another amount'):
            Budget(tmp_path, 1, extend_seconds=5, extension_id='one')


def boxes(player_y, ball_y, present=True):
    return ObjectFrame((DetectedObject('player', True, (140., player_y, 4., 8.)),
                        DetectedObject('ball', present, (80., ball_y, 4., 8.) if present else None),
                        DetectedObject('enemy', True, (10., 100., 4., 8.))))


@pytest.mark.parametrize('relative,expected', [('less', 3), ('same', 0), ('more', 2)])
def test_both_vertical_conventions(relative, expected):
    actions = ToyEnvironment.actions
    facts = [Fact('logical', True, 'present', name+'_t') for name in ('player', 'ball')]
    facts.append(Fact('comparative', relative, 'y', 'player_t', 'ball_t'))
    assert ReactivePolicy('RIGHT', 'LEFT', actions).fact_action(facts) == expected
    inverse = {'less': 'more', 'more': 'less', 'same': 'same'}[relative]
    visual = [Fact('comparative', inverse, 'y', 'player_t', 'ball_t')]
    assert ReactivePolicy('RIGHT', 'LEFT', actions, 'visual').fact_action(visual) == expected


def test_availability_is_not_false_presence():
    adapter = FactAdapter('visual')
    assert adapter.observe([]).cause == 'empty_state'
    assert adapter.observe([Fact('comparative', 'same', 'x', 'player_t', 'ball_t')]).cause == 'relation_unavailable'
    assert adapter.observe([Fact('comparative', 'more', 'y', 'player_t', 'ball_t')]).cause == 'insufficient_history'
    missing = FactAdapter().observe([Fact('logical', False, 'present', 'ball_t')])
    assert missing.cause == 'observed_absence'


def test_oc_objects_and_facts_same_history_reset_points_and_recovery():
    relations = PongRelationsV1()
    objects = ReactivePolicy('RIGHT', 'LEFT', ToyEnvironment.actions)
    facts = ReactivePolicy('RIGHT', 'LEFT', ToyEnvironment.actions)
    frames = [boxes(80, 110), boxes(80, 110)]*6 + [boxes(80, 110, False), boxes(110, 80), boxes(80, 80)]
    for i, frame in enumerate(frames):
        reward = 1. if i == 5 else 0.
        if reward:
            objects.reset()
            facts.reset()
        assert objects.object_action(frame) == facts.fact_action(relations.transform(frame, raw_reward=reward))
    objects.reset()
    facts.reset()
    relations.reset()
    assert objects.object_action(boxes(80, 80)) == facts.fact_action(relations.transform(boxes(80, 80)))


def test_assisted_gate_requires_real_positive_points_and_full_pairing():
    report = {'policies': {'object_reactive': {600001: {'status': 'completed', 'points_won': 0}}},
              'paired_vs_random': {'object_reactive': {'mean_difference': 10, 'paired_seeds': [600001]}}}
    assert not assisted_gate(report, [600001])['enabled']
    report['policies']['object_reactive'][600001]['points_won'] = 1
    assert assisted_gate(report, [600001])['enabled']


def test_reports_refresh_after_partial_episode_completes(tmp_path):
    from experiments.evaluate_pong_heuristics import episode_identity, play, report
    source = tmp_path/'pilot'
    source.mkdir()
    (source/'protocol.json').write_text(json.dumps({'protocol': {'environments': {'ocatari': {}}}}))
    manifest = {'config': {'source_campaign': str(source), 'horizon': 20, 'smoke_horizon': 2},
                'pilot': {'pilot_protocol_sha256': 'pilot'}, 'runtime': {}, 'source_hashes': {}}
    directions = {'up': 'RIGHT', 'down': 'LEFT'}
    root = tmp_path/'results'
    seed = 600001
    for policy in ('random', 'noop'):
        expected = episode_identity(manifest, 'ocatari', policy, seed, False, directions)
        play(ToyEnvironment, 'ocatari', policy, seed, 20, 60, directions,
             folder=root/policy/str(seed), expected=expected)
    initial = report(root, 'ocatari', [seed], manifest=manifest, directions=directions)
    assert initial['paired_vs_random']['noop']['paired_seeds'] == [seed]
    expected = episode_identity(manifest, 'ocatari', 'fact_reactive', seed, False, directions)
    # A partial persisted unit must not enter paired comparisons.
    from experiments.evaluate_pong_heuristics import HeuristicController
    episode(ToyEnvironment, HeuristicController('fact_reactive', 'ocatari', directions),
            root/'fact_reactive'/str(seed), expected, seed=seed, horizon=20, seconds=60,
            rng_seed=seed, decision_limit=2)
    assert 'fact_reactive' not in report(root, 'ocatari', [seed], manifest=manifest, directions=directions)['paired_vs_random']
    play(ToyEnvironment, 'ocatari', 'fact_reactive', seed, 20, 60, directions,
         folder=root/'fact_reactive'/str(seed), expected=expected)
    updated = report(root, 'ocatari', [seed], manifest=manifest, directions=directions)
    assert updated['paired_vs_random']['fact_reactive']['paired_seeds'] == [seed]
