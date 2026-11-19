import copy
import json
from pathlib import Path

import pytest

from experiments import confirm_rrtl_anova as confirmation
from experiments import diagnose_rrtl_improvements as original
from bayesian_rrtl.online.protocol import resolve

ROOT=Path(__file__).resolve().parents[1]


def design():
    return json.loads((ROOT/'configs/breakout_anova_confirmation_20261109.json').read_text())


def test_distinct_schema_keeps_original_full_comparison_strict():
    cfg=design();confirmation.validate_config(cfg)
    with pytest.raises(ValueError):original.validate_config(cfg)
    old=json.loads((ROOT/'configs/breakout_rrtl_improvements_20261108.json').read_text())
    original.validate_config(old)
    with pytest.raises(ValueError):confirmation.validate_config(old)


def test_confirmation_uses_fresh_seeds_and_preserves_maintenance_cadence():
    cfg=design();parents=json.loads((ROOT/'configs/breakout_confirmation_parents_20261109.json').read_text())
    resolved=resolve(parents)
    assert len(resolved['cells'])==5
    assert {c['seed'] for c in resolved['cells']}==set(cfg['seeds'])
    assert all(c['estimator']['split_test']=='anova' for c in resolved['cells'])
    assert cfg['evaluation_interval']==25000  # Also the unchanged refit cadence.
    assert cfg['continuation_decisions']==500000
    assert len(cfg['seeds'])*len(cfg['modes'])*cfg['continuation_decisions']==7500000
    assert all(c['environment']['horizon']==27000 for c in resolved['cells'])
    for key in ['eta_q','bootstrap','repair']:
        old=json.loads((ROOT/'configs/breakout_rrtl_improvements_20261108.json').read_text())
        assert all(cfg['variants'][m][key]==old['variants'][m][key] for m in cfg['modes'])


@pytest.mark.parametrize('key,value',[('seeds',[710602]),('continuation_decisions',100000),('evaluation_interval',50000)])
def test_no_silent_reuse_of_old_seeds_or_shorter_followup(key,value):
    cfg=copy.deepcopy(design());cfg[key]=value
    with pytest.raises(ValueError):confirmation.validate_config(cfg)
