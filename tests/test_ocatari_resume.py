"""Step 4: real boundary matrix, process isolation and historical snapshots."""
import copy
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from bayesian_rrtl.agent import BayesianQAgent
from bayesian_rrtl.atari import AtariConfig, AtariRelationalEnvironment
from bayesian_rrtl.checkpoint import load_checkpoint
from bayesian_rrtl.game_registry import get_game_spec
from bayesian_rrtl.q_config import BayesianQConfig
from bayesian_rrtl.training import QTrainingSession
from experiments.validate_ocatari_resume import validate_resume

pytestmark = [pytest.mark.atari, pytest.mark.ocatari]


def test_boundary_matrix_matches_original_128_steps_in_separate_process(tmp_path):
    report = validate_resume(tmp_path / 'matrix')
    assert report['status'] == 'passed' and len(report['cases']) == 8
    for name, case in report['cases'].items():
        assert case['compared_steps'] == (1 if name == 'before_truncation' else 128)
        assert case['max_deviation'] == 0


def test_h5_process_resume_matches_full_future_state(tmp_path):
    c = AtariConfig(game='Pong', extractor='ocatari_ram_v1', include_incomplete_states=True,
                    max_episode_steps=80)
    q = BayesianQConfig(actions=get_game_spec('Pong', c.extractor).actions, seed=41,
                        gamma=.99, reward_min=-1., reward_max=1., m=1, burn_in=2, draws=4,
                        fit_interval=8, batch_size_per_action=8)
    env = AtariRelationalEnvironment(c)
    try:
        session = QTrainingSession(BayesianQAgent(q, env.encoder()), env)
        session.run(19)
        checkpoint = session.save(tmp_path / 'initial.rrtl')
        session.run(128)
        output = tmp_path / 'child'
        subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1] / 'train_atari.py'),
                        '--resume', str(checkpoint), '--output', str(output), '--steps', '128',
                        '--eval-split', 'none'], check=True, capture_output=True, text=True)
        child, collector = load_checkpoint(output / 'final_checkpoint.rrtl')
        assert collector['history'] == session.history
        assert collector['state'] == session.state
        for key in ('episode_id', 'step_id', 'episode_return', 'raw_episode_return', 'done'):
            assert collector[key] == getattr(session, key)
        for key in ('environment_rng', 'exploration_rng', 'replay_rng'):
            assert getattr(child, key).bit_generator.state == getattr(session.agent, key).bit_generator.state
        for a, b in zip(child.mcmc_rngs, session.agent.mcmc_rngs):
            assert a.bit_generator.state == b.bit_generator.state
        assert child.training_seeds == session.agent.training_seeds
        assert child.replay.next_id == session.agent.replay.next_id == 147
        for a, b in zip(child.replay.transitions, session.agent.replay.transitions):
            for key in ('transition_id', 'action', 'reward', 'raw_reward', 'terminated', 'truncated', 'episode_id', 'step_id'):
                assert getattr(a, key) == getattr(b, key)
            for key in ('state', 'next_state'):
                np.testing.assert_array_equal(getattr(a, key).values, getattr(b, key).values)
                np.testing.assert_array_equal(getattr(a, key).observed, getattr(b, key).observed)
        for name in ('model', 'target_model'):
            a, b = getattr(child, name), getattr(session.agent, name)
            assert a.model_id == b.model_id
            for pa, pb in zip(a.posteriors, b.posteriors):
                if pa is not None:
                    assert pa.draws == pb.draws and pa.trace == pb.trace
                    assert pa.sigma2 == pb.sigma2
        for a, b in zip(child.last_fit.datasets, session.agent.last_fit.datasets):
            assert a.data_hash == b.data_hash and a.transition_ids == b.transition_ids
            np.testing.assert_array_equal(a.targets, b.targets)
    finally:
        env.close()


@pytest.mark.parametrize('game', ['Breakout', 'Pong', 'DemonAttack'])
def test_v1_old_config_shape_restores_with_explicit_resolved_defaults(game):
    c = AtariConfig(game=game, extractor=f'legacy_visual_{game.lower()}_v1', max_episode_steps=20)
    left, right = AtariRelationalEnvironment(c), AtariRelationalEnvironment(AtariConfig(**c.resolved_dict()))
    try:
        left.reset(seed=7)
        left.step(1)
        old = left.snapshot()
        assert old['kind'] == 'atari_relational_v1'
        assert not {'env_id', 'relation_schema', 'reset_policy', 'reward_transform'} & old['config'].keys()
        right.restore(old)
        for action in [1, 2, 3, 0] * 5:
            a, b = left.step(action), right.step(action)
            assert a == b
            if a[-2] or a[-1]:
                break
    finally:
        left.close()
        right.close()


@pytest.mark.parametrize('corruption', ['versions', 'source_hashes', 'rom_sha256', 'actions', 'missing',
                                       'wrapper_fields', 'wrapper_config', 'done', 'relation_history'])
def test_incompatible_snapshot_rejected_before_restore_mutates_state(corruption):
    c = AtariConfig(game='Pong', extractor='ocatari_ram_v1', include_incomplete_states=True)
    env = AtariRelationalEnvironment(c)
    try:
        env.reset(seed=41)
        original = env.snapshot()
        saved = copy.deepcopy(original)
        backend = saved['backend']
        if corruption in ('versions', 'source_hashes', 'rom_sha256', 'actions'):
            backend['manifest'][corruption] = 'wrong'
        elif corruption == 'missing':
            del backend['rng']
        elif corruption == 'wrapper_fields':
            del backend['wrappers'][0][1]['_elapsed_steps']
        elif corruption == 'wrapper_config':
            backend['wrappers'][0][1]['_max_episode_steps'] = 2
        elif corruption == 'done':
            saved['done'] = not backend['done']
        else:
            saved['relations']['previous'] = None
        with pytest.raises(ValueError, match='mismatch'):
            env.restore(saved)
        assert env.snapshot() == original
    finally:
        env.close()
