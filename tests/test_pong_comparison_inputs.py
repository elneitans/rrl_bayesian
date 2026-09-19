"""Gate B requires real ALE and optional OCAtari; no silent skips."""

import pytest

from experiments.validate_pong_comparison_inputs import validate_inputs


@pytest.mark.atari
@pytest.mark.parametrize('backend', [
    'legacy_visual_pong_v1', pytest.param('ocatari_ram_v1', marks=pytest.mark.ocatari),
])
def test_real_fixed_action_inputs_points_truncations_resets(backend):
    result = validate_inputs(backend)
    assert result['status'] == 'passed'
    assert result['semantic_inputs_equal'] and result['reward_raw_flags_logs_equal']
    assert result['environment_step_calls'] == [1030, 1030]
    assert result['nonterminal_points'] > 0 and result['truncations'] >= 2 and result['resets'] >= 3
