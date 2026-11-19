from collections import Counter

import numpy as np

from experiments.breakout_bart_replay_coverage import cover_batch, training_signature


def test_cover_batch_retains_size_and_changes_only_required_rows():
    available = {0: 'initial_serve', 1: 'initial_serve',
                 2: 'after_life_loss', 3: 'after_life_loss',
                 **dict.fromkeys(range(4, 18), 'in_play')}
    original = [0, *range(4, 13)]
    quota = {'initial_serve': 2, 'after_life_loss': 2, 'in_play': 5}
    covered, details = cover_batch(original, available, quota, np.random.default_rng(11))
    counts = Counter(available[i] for i in covered)
    assert counts == {'initial_serve': 2, 'after_life_loss': 2, 'in_play': 6}
    assert len(covered) == len(set(covered)) == len(original)
    assert details['substitutions'] == len(set(covered) - set(original)) == 3


def test_cover_batch_reports_a_floor_that_is_unattainable():
    available = {7: 'after_life_loss', **dict.fromkeys(range(8, 25), 'in_play')}
    quota = {'initial_serve': 1, 'after_life_loss': 2, 'in_play': 5}
    covered, details = cover_batch(list(range(8, 16)), available, quota, np.random.default_rng(7))
    assert 7 in covered
    assert details['attainable'] == {'initial_serve': 0, 'after_life_loss': 1, 'in_play': 5}
    assert details['substitutions'] == 1


def test_control_prefix_signature_tracks_data_and_ignores_wall_time():
    original = {'interaction': 512, 'episode_id': 1, 'step_id': 3, 'episode_seed': 42,
                'action': 1, 'raw_reward': 0, 'terminated': False, 'truncated': False,
                'fit_id': 1, 'fit': {'target_model_id': 0,
                'actions': {'FIRE': {'data_hash': 'a', 'elapsed_seconds': 12.0}}}}
    same = {**original, 'fit': {'target_model_id': 0,
            'actions': {'FIRE': {'data_hash': 'a', 'elapsed_seconds': 24.0}}}}
    other = {**same, 'fit': {'target_model_id': 0,
             'actions': {'FIRE': {'data_hash': 'b', 'elapsed_seconds': 24.0}}}}
    assert training_signature(original) == training_signature(same)
    assert training_signature(original) != training_signature(other)
