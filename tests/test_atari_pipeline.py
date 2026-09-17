"""H6: real ALE, replaceable object boundary, snapshots and paired CLI."""

import json
from pathlib import Path
import pickle
import subprocess
import sys

import numpy as np
import pytest

from bayesian_rrtl.agent import BayesianQAgent
from bayesian_rrtl.atari import AtariConfig, AtariRelationalEnvironment
from bayesian_rrtl.objects import BreakoutRelations, DetectedObject, ObjectFrame, make_extractor
from bayesian_rrtl.q_config import BayesianQConfig
from bayesian_rrtl.training import QTrainingSession
from preprocessing import BreakoutPreprocessor, get_state_comparative_breakout, get_state_logical_breakout


def test_object_contract_missing_coordinates_and_future_extractor():
    missing = DetectedObject('ball', False)
    assert missing.center is None
    assert DetectedObject('ball', True, (10, 20, 2, 4)).center == (11, 22)
    with pytest.raises(ValueError, match='stale'):
        DetectedObject('ball', False, (10, 20, 2, 4))
    with pytest.raises(ValueError, match='Duplicate'):
        ObjectFrame((missing, missing))
    with pytest.raises(NotImplementedError, match='deferred'):
        make_extractor('ocatari_ram')
    with pytest.raises(ValueError, match='OCAtari'):
        AtariConfig(extractor='ocatari_ram')


@pytest.mark.parametrize('representation', ['comparative', 'logical'])
def test_object_order_presence_temporal_reset_and_catalog(representation):
    relations = BreakoutRelations(representation, True)
    frame = ObjectFrame((DetectedObject('ball', False), DetectedObject('player', True, (80, 189, 16, 4))))
    first = relations.transform(frame)
    relations.transform(ObjectFrame((DetectedObject('ball', True, (90, 100, 2, 4)), frame.objects[1])))
    relations.reset()
    assert relations.transform(ObjectFrame(tuple(reversed(frame.objects)))) == first
    batch = relations.encoder().transform([first, []])
    assert len(batch) == 2 and not batch.observed[1].any()
    assert any(f.obj1 == 'ball_t' and f.name == 'present' and f.value == 'False' for f in first)


@pytest.mark.atari
@pytest.mark.parametrize('representation', ['comparative', 'logical'])
@pytest.mark.parametrize('incomplete', [True, False])
def test_real_frames_keep_legacy_facts(representation, incomplete):
    config = AtariConfig(representation=representation, include_incomplete_states=incomplete, max_episode_steps=40)
    adapter = AtariRelationalEnvironment(config)
    legacy = BreakoutPreprocessor()
    builder = get_state_comparative_breakout if representation == 'comparative' else get_state_logical_breakout
    previous = None
    try:
        observation, _ = adapter.env.reset(seed=125)
        adapter.relations.reset()
        for _ in range(30):
            actual = adapter._state(observation)
            info = legacy.get_info(observation)
            assert actual == builder(info, previous, incomplete)
            adapter.encoder().transform([actual])  # static catalog includes ALL emitted facts
            previous = info
            observation, _, terminal, truncated, _ = adapter.env.step(1)
            if terminal or truncated:
                break
    finally:
        adapter.close()


@pytest.mark.atari
def test_ale_resume_sticky_rng_temporal_history_and_truncation(tmp_path):
    config = AtariConfig(max_episode_steps=19, repeat_action_probability=0.25, include_incomplete_states=True)
    left, right = AtariRelationalEnvironment(config), AtariRelationalEnvironment(config)
    try:
        left.reset(seed=31)
        for action in (1, 3, 2, 1):
            left.step(action)
        snapshot = pickle.loads(pickle.dumps(left.snapshot()))
        right.restore(snapshot)
        seen_truncation = False
        for action in (3, 3, 2, 1, 0) * 5:
            a, b = left.step(action), right.step(action)
            assert a == b
            if a[-1]:
                seen_truncation = True
            if a[-2] or a[-1]:
                break
        assert seen_truncation
        assert left.env.unwrapped.ale.cloneSystemState() == right.env.unwrapped.ale.cloneSystemState()
        assert left.relations.previous == right.relations.previous
    finally:
        left.close()
        right.close()


@pytest.mark.atari
def test_full_atari_checkpoint_replays_next_fits(tmp_path):
    env_config = AtariConfig(max_episode_steps=30, include_incomplete_states=True)
    config = BayesianQConfig(actions=AtariRelationalEnvironment.actions, gamma=0.99, reward_min=-1,
                              reward_max=1, fit_interval=8, m=1, burn_in=2, draws=4,
                              batch_size_per_action=8, resample_repeated_actions=True)
    left_env, right_env = AtariRelationalEnvironment(env_config), AtariRelationalEnvironment(env_config)
    try:
        left = QTrainingSession(BayesianQAgent(config, left_env.encoder()), left_env)
        left.run(11)
        checkpoint = left.save(tmp_path / 'checkpoint.rrtl')
        right = QTrainingSession.load(checkpoint, right_env)
        left.run(16)
        right.run(16)
        assert left.history == right.history
        assert left.agent.exploration_rng.bit_generator.state == right.agent.exploration_rng.bit_generator.state
        assert [r.bit_generator.state for r in left.agent.mcmc_rngs] == [r.bit_generator.state for r in right.agent.mcmc_rngs]
        for a, b in zip(left.agent.model.posteriors, right.agent.model.posteriors):
            if a is not None:
                assert a.draws == b.draws and a.trace == b.trace
        for a, b in zip(left.agent.last_fit.datasets, right.agent.last_fit.datasets):
            assert a.data_hash == b.data_hash
            np.testing.assert_array_equal(a.targets, b.targets)
    finally:
        left_env.close()
        right_env.close()


def test_warmup_blocks_fit_even_when_interval_elapsed():
    from bayesian_rrtl.toy_mdp import TwoStepMDP
    agent = BayesianQAgent(BayesianQConfig(fit_interval=2, warmup_interactions=5, m=1, burn_in=1, draws=2),
                           TwoStepMDP.encoder())
    session = QTrainingSession(agent, TwoStepMDP())
    session.run(4)
    assert agent.model.model_id == 0
    session.run(1)
    assert agent.model.model_id == 1


@pytest.mark.atari
def test_runner_pair_resume_evaluation_and_profile(tmp_path):
    root = Path(__file__).resolve().parents[1]
    config = {'environment': AtariConfig(max_episode_steps=12).to_dict(),
              'agent': BayesianQConfig(actions=AtariRelationalEnvironment.actions, gamma=0.99,
                  reward_min=-1, reward_max=1, m=1, burn_in=2, draws=4, fit_interval=8,
                  batch_size_per_action=8, validation_seeds=(100001,)).to_dict(), 'interactions': 16}
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(config))
    def run(*args):
        result = subprocess.run([sys.executable, str(root / 'train_atari.py'), *map(str, args)],
                                cwd=tmp_path, capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, result.stdout + result.stderr
    for variant in ('bayesian', 'corrected'):
        run('--config', path, '--variant', variant, '--output', tmp_path / variant)
        summary = json.loads((tmp_path / variant / 'summary.json').read_text())
        assert summary['interactions'] == 16 and summary['finite_data']
        assert summary['profile']['process_peak_rss_bytes'] > 0
        assert not (tmp_path / variant / 'test.json').exists()
    summaries = [json.loads((tmp_path / v / 'summary.json').read_text()) for v in ('bayesian', 'corrected')]
    assert summaries[0]['training_seeds'][0] == summaries[1]['training_seeds'][0]
    checkpoint = tmp_path / 'bayesian/final_checkpoint.rrtl'
    saved = checkpoint.read_bytes()
    run('--resume', checkpoint, '--output', tmp_path / 'resumed', '--steps', 7)
    assert json.loads((tmp_path / 'resumed/summary.json').read_text())['interactions'] == 23
    run('--checkpoint', checkpoint, '--eval-split', 'test')
    assert checkpoint.read_bytes() == saved
    assert (tmp_path / 'bayesian/test.json').is_file()


def test_h5_checkpoint_before_warmup_option_remains_loadable(tmp_path):
    import hashlib
    from bayesian_rrtl.checkpoint import MAGIC, load_checkpoint, save_checkpoint
    from bayesian_rrtl.toy_mdp import TwoStepMDP
    agent = BayesianQAgent(BayesianQConfig(), TwoStepMDP.encoder())
    path = save_checkpoint(agent, tmp_path / 'old.rrtl')
    payload = pickle.loads(path.read_bytes().split(b'\n', 2)[2])
    del payload['config']['warmup_interactions']
    payload['config_hash'] = hashlib.sha256(json.dumps(payload['config'], sort_keys=True).encode()).hexdigest()
    data = pickle.dumps(payload)
    path.write_bytes(MAGIC + hashlib.sha256(data).hexdigest().encode() + b'\n' + data)
    restored, _ = load_checkpoint(path)
    assert restored.config.warmup_interactions == 0
    assert restored.exploration_rng.bit_generator.state == agent.exploration_rng.bit_generator.state


def test_offline_diagnostics_use_frozen_dataset_and_leave_checkpoint_intact(tmp_path):
    from bayesian_rrtl.checkpoint import save_checkpoint
    from bayesian_rrtl.toy_mdp import TwoStepMDP
    from experiments.diagnose_atari import diagnose
    agent = BayesianQAgent(BayesianQConfig(m=1, fit_interval=8, burn_in=2, draws=6), TwoStepMDP.encoder())
    QTrainingSession(agent, TwoStepMDP()).run(16)
    checkpoint = save_checkpoint(agent, tmp_path / 'agent.rrtl')
    original = checkpoint.read_bytes()
    report = diagnose(checkpoint, tmp_path / 'diagnostics', chains=2, burn_in=2, draws=6)
    assert report['fit_id'] == agent.last_fit.fit_id
    for dataset in agent.last_fit.datasets:
        name = agent.config.actions[dataset.action]
        assert report['actions'][name]['dataset_hash'] == dataset.data_hash
        stored = np.load(tmp_path / 'diagnostics' / f'{name}_draws.npz')
        np.testing.assert_array_equal(stored['targets'], dataset.targets)
        np.testing.assert_array_equal(stored['transition_ids'], dataset.transition_ids)
        assert stored['predictions'].shape[:2] == (2, 6)
    assert checkpoint.read_bytes() == original
