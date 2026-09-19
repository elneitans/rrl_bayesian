import copy
from dataclasses import replace
import hashlib
import json
import pickle

import pytest

from bayesian_rrtl import baseline_checkpoint as checkpoint
from bayesian_rrtl.baseline_adapter import BaselineLearnerConfig, RelationalBaselineAgent
from bayesian_rrtl.baseline_training import BaselineTrainingSession
from bayesian_rrtl.checkpoint import load_checkpoint as load_h5
from bayesian_rrtl.comparison import ComparisonPolicyConfig
from experiments.validate_baseline_resume import (SplitSignalEnvironment, digest, fingerprint, validate_case)


def session():
    env = SplitSignalEnvironment()
    agent = RelationalBaselineAgent(env.actions, env.encoder(),
        BaselineLearnerConfig(eta_q=1., gamma=0., min_sample_size=6, significance_level=.05))
    return BaselineTrainingSession(agent, env, environment_factory=SplitSignalEnvironment)


def read_payload(path):
    return pickle.loads(path.read_bytes().split(b'\n', 2)[2])


def rewrite(path, payload):
    # Recompute checksums deliberately, to exercise semantic validation as well.
    payload['contract_hash'] = checkpoint._hash(payload['contract'])
    data = pickle.dumps(payload)
    path.write_bytes(checkpoint.MAGIC + hashlib.sha256(data).hexdigest().encode() + b'\n' + data)


class RestoreSpy(SplitSignalEnvironment):
    restored = False
    closed = False

    def restore(self, snapshot):
        self.restored = True
        super().restore(snapshot)

    def close(self):
        self.closed = True


def test_save_load_preserves_learner_collector_and_checkpoint_is_independent(tmp_path):
    left = session()
    left.run(12)
    before = digest(fingerprint(left))
    path = left.save(tmp_path / 'progress.rrtlc')
    assert digest(fingerprint(left)) == before
    saved_bytes = path.read_bytes()
    right = BaselineTrainingSession.load(path, environment_factory=SplitSignalEnvironment)
    assert digest(fingerprint(right)) == before
    assert right.agent.n_splits == 1
    assert right.agent.root_nodes['NOOP'] is not left.agent.root_nodes['NOOP']
    for _ in range(128):
        assert left.step() == right.step()
        assert digest(fingerprint(left)) == digest(fingerprint(right))
    assert left.agent.n_splits == right.agent.n_splits == 3
    assert path.read_bytes() == saved_bytes


def test_empty_collector_and_explicit_order_roundtrip(tmp_path):
    original = session()
    original.agent = RelationalBaselineAgent(original.agent.actions, original.agent.encoder, original.agent.config,
                                             ordered_catalog=tuple(reversed(original.agent.catalog)))
    path = original.save(tmp_path / 'initial.rrtlc')
    loaded = BaselineTrainingSession.load(path, environment_factory=SplitSignalEnvironment)
    assert loaded.agent.catalog == original.agent.catalog
    assert digest(fingerprint(original)) == digest(fingerprint(loaded))
    assert loaded.step() == original.step()


@pytest.mark.parametrize('change', [
    'schema', 'learner', 'source', 'numpy_version', 'catalog_order', 'literal_list', 'actions',
    'interaction', 'split_counter', 'statistic_n', 'missing_statistics', 'tree_parent', 'tree_q',
    'episode_return', 'step_counter', 'history_id', 'history_epsilon', 'episode_rng', 'policy_rng',
    'episode_seed', 'environment_done', 'environment_kind', 'environment_semantics', 'snapshot_index',
])
def test_corrupt_semantic_state_rejected_before_restore(tmp_path, change):
    original = session()
    original.run(12)
    path = original.save(tmp_path / 'agent.rrtlc')
    payload = read_payload(path)
    root = payload['learner_state']['roots']['NOOP']
    contract = payload['contract']
    collector = payload['collector']
    if change == 'schema':
        payload['schema_version'] = 99
    elif change == 'learner':
        payload['learner'] = 'BayesianQAgent'
    elif change == 'source':
        contract['runtime']['sources']['bayesian_rrtl/baseline.py'] = 'different'
    elif change == 'numpy_version':
        contract['runtime']['packages']['numpy'] = 'different'
    elif change == 'catalog_order':
        contract['learner']['ordered_catalog'].reverse()
    elif change == 'literal_list':
        contract['learner']['ordered_literals'] = []
    elif change == 'actions':
        contract['learner']['actions'] = ['wrong']
    elif change == 'interaction':
        payload['learner_state']['interactions'] += 1
    elif change == 'split_counter':
        payload['learner_state']['n_splits'] += 1
    elif change == 'statistic_n':
        root.refinements_stats[root.refinements[0]]['Total']['n'] += 1
    elif change == 'missing_statistics':
        root.refinements_stats = {}
    elif change == 'tree_parent':
        root.children[0].parent = None
    elif change == 'tree_q':
        root.q_value = float('nan')
    elif change == 'episode_return':
        collector['episode_return'] += 1
    elif change == 'step_counter':
        collector['step_id'] += 1
    elif change == 'history_id':
        collector['history'][0]['transition_id'] = 10
    elif change == 'history_epsilon':
        collector['history'][0]['epsilon'] = .25
    elif change == 'episode_rng':
        collector['environment_rng'] = collector['exploration_rng']
    elif change == 'policy_rng':
        collector['exploration_rng'] = {'invalid': True}
    elif change == 'episode_seed':
        collector['training_seeds'][0] = 100001
    elif change == 'environment_done':
        payload['environment_snapshot']['done'] = True
    elif change == 'environment_kind':
        payload['environment_snapshot']['kind'] = 'wrong'
    elif change == 'environment_semantics':
        contract['environment']['runtime']['environment'] = 'different'
    elif change == 'snapshot_index':
        payload['environment_snapshot']['index'] = -1
    rewrite(path, payload)
    env = RestoreSpy()
    before = env.snapshot()
    with pytest.raises(ValueError):
        BaselineTrainingSession.load(path, env)
    assert not env.restored and env.snapshot() == before and not env.closed


def test_checksum_and_old_formats_are_rejected_without_unpickling(tmp_path, monkeypatch):
    path = session().save(tmp_path / 'agent.rrtlc')
    saved = path.read_bytes()
    def forbidden(*args, **kwargs):
        pytest.fail('Format and checksum must be checked before pickle')
    monkeypatch.setattr(checkpoint.pickle, 'loads', forbidden)
    for data in (saved[:-1], b'RRTL-B-CHECKPOINT-1\n' + saved, b'old H0 pickle'):
        path.write_bytes(data)
        with pytest.raises(ValueError):
            BaselineTrainingSession.load(path, SplitSignalEnvironment())
    path.write_bytes(saved)
    with pytest.raises(ValueError, match='format'):
        load_h5(path)


def test_expected_contracts_and_atomic_write_failure(tmp_path, monkeypatch):
    original = session()
    original.run(12)
    path = original.save(tmp_path / 'progress.rrtlc')
    data = path.read_bytes()
    for options in ({'expected_config': replace(original.agent.config, eta_q=.5)},
                    {'expected_policy': ComparisonPolicyConfig(validation_seeds=(42,), test_seeds=(43,))}):
        env = RestoreSpy()
        with pytest.raises(ValueError, match='Expected'):
            BaselineTrainingSession.load(path, env, **options)
        assert not env.restored
    original.run(1)
    def fail_replace(*args):
        raise OSError('simulated interrupted publication')
    monkeypatch.setattr(checkpoint.os, 'replace', fail_replace)
    with pytest.raises(OSError, match='interrupted'):
        original.save(path)
    assert path.read_bytes() == data
    assert list(tmp_path.iterdir()) == [path]


def test_saves_only_between_completed_updates(tmp_path, monkeypatch):
    original = session()
    path = original.save(tmp_path / 'initial.rrtlc')
    data = path.read_bytes()
    observe = original.agent.observe
    def wrapped(*args, **kwargs):
        with pytest.raises(ValueError, match='completed'):
            original.save(path)
        return observe(*args, **kwargs)
    monkeypatch.setattr(original.agent, 'observe', wrapped)
    original.step()
    assert path.read_bytes() == data
    original.save(path)
    def fail(*args, **kwargs):
        raise RuntimeError('simulated interrupted update')
    monkeypatch.setattr(original.agent, 'observe', fail)
    with pytest.raises(RuntimeError):
        original.step()
    with pytest.raises(ValueError, match='completed'):
        original.save(path)
    with pytest.raises(ValueError, match='Incomplete'):
        original.step()
    assert BaselineTrainingSession.load(path, SplitSignalEnvironment()).interactions == 1


def test_artifact_evaluation_does_not_use_modified_live_agent(tmp_path):
    original = session()
    original.run(12)
    path = original.save(tmp_path / 'final.rrtlc')
    data = path.read_bytes()
    original.agent.root_nodes['NOOP'].q_value = 9999.
    loaded = BaselineTrainingSession.load(path, environment_factory=SplitSignalEnvironment)
    before = copy.deepcopy(fingerprint(loaded))
    expected = loaded.evaluate((100001,), horizon=16)
    assert fingerprint(loaded) == before
    actual = checkpoint.evaluate_baseline_checkpoint(path, (100001,), horizon=16,
                                                     environment_factory=SplitSignalEnvironment)
    assert actual == expected and path.read_bytes() == data
    assert loaded.agent.root_nodes['NOOP'].q_value != 9999.


@pytest.mark.parametrize('backend', [
    'synthetic', pytest.param('legacy_visual_pong_v1', marks=pytest.mark.atari),
    pytest.param('ocatari_ram_v1', marks=[pytest.mark.atari, pytest.mark.ocatari]),
])
def test_gate_c_resume_between_processes_and_final_evaluation(tmp_path, backend):
    report = validate_case(backend, tmp_path / backend)
    assert report['status'] == 'passed' and report['continuation_decisions'] == 128
    assert report['local_and_process_exact'] and report['evaluation_checkpoint_unchanged']
    if backend == 'synthetic':
        assert (report['splits_before'], report['splits_after']) == (1, 3)
    assert json.loads((tmp_path / backend / 'report.json').read_text()) == report
