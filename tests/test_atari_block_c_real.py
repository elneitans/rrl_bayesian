"""Bounded opt-in pinned-runtime C13/C14 validation. No reserved test seeds."""
import json
import os
from pathlib import Path

import pytest

from bayesian_rrtl.online.environments import GameEnvironmentConfig
from experiments.validate_atari_games import validate

pytestmark=[pytest.mark.atari,pytest.mark.ocatari]


@pytest.mark.parametrize('game',['breakout','demonattack'])
def test_real_detector_and_relations_continue_in_another_process(tmp_path,game):
    if os.environ.get('RRL_C_REAL_TEST')!='1':
        pytest.skip('Opt-in pinned OCAtari/Gym/ALE runtime required')
    config=GameEnvironmentConfig(**json.loads((Path(__file__).resolve().parents[1]/'configs'/f'{game}_common_v1.json').read_text()))
    result=validate(config,tmp_path/game)
    assert result['exact_resume_between_processes'] and result['compared_steps']==64


@pytest.mark.parametrize('game',['Breakout','DemonAttack'])
def test_real_verified_actions_and_explicit_fire(game):
    if os.environ.get('RRL_C_REAL_TEST')!='1':
        pytest.skip('Opt-in pinned OCAtari/Gym/ALE runtime required')
    from bayesian_rrtl.online.environments import CommonGameFactory
    config=GameEnvironmentConfig(**json.loads((Path(__file__).resolve().parents[1]/'configs'/f'{game.lower()}_common_v1.json').read_text()))
    env=CommonGameFactory(config)()
    try:
        assert tuple(env.backend.actions)==tuple(config.spec().actions)
        positions={}
        for name in ('NOOP','RIGHT','LEFT','FIRE'):
            env.reset(seed=640001)
            before=next(o.center[0] for o in env.backend.current_frame.objects if o.identity=='player')
            for _ in range(8):
                env.step(env.actions.index(name))
            player=next(o.center[0] for o in env.backend.current_frame.objects if o.identity=='player')
            positions[name]=player-before
        assert positions['RIGHT']>positions['NOOP']>positions['LEFT']
        env.reset(seed=640001)
        for _ in range(8):
            env.step(env.actions.index('FIRE'))
        obj=next(o for o in env.backend.current_frame.objects if o.identity==('ball' if game=='Breakout' else 'player_missile'))
        if game=='Breakout':
            assert obj.present
        else:
            assert obj.center[1]<170  # observed shot displacement, not a presence inference
    finally:
        env.close()
