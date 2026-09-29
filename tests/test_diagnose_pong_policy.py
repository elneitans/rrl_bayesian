import json
import pickle

import numpy as np
import pytest

from bayesian_rrtl.encoding import RelationKey, RelationalEncoder
from preprocessing import Fact
from experiments.diagnose_pong_policy import trace_episode


class Env:
    actions = ('left', 'right')

    def reset(self, *, seed):
        self.i = 0
        return [Fact('comparative', 'less', 'y', 'player', 'ball')]

    def step(self, action):
        self.i += 1
        return self.reset_state(), -1., -1., self.i == 3, False

    def reset_state(self):
        return [Fact('comparative', 'less', 'y', 'player', 'ball')]


class Agent:
    def __init__(self):
        self.encoder = RelationalEncoder([RelationKey('comparative', 'y', 'player', 'ball')])
        self.model = Model()

    def encode_state(self, state):
        return self.encoder.transform([state])


class Model:
    def predict(self, batch):
        return np.array([[1., 2.]])


def test_trace_records_preaction_facts_q_and_terminal_without_training(tmp_path):
    agent = Agent()
    before = pickle.dumps(agent)
    r = trace_episode(agent, Env(), 320101, 10, tmp_path / 'trace')
    assert pickle.dumps(agent) == before
    assert r['stop_reason'] == 'terminated' and r['steps'] == 3
    assert r['actions'] == {'right': 3} and r['longest_same_action_streak'] == 3
    rows = [json.loads(s) for s in (tmp_path / 'trace/trace.jsonl').read_text().splitlines()]
    assert rows[0]['facts'][0]['value'] == 'less'
    assert rows[0]['q'] == {'left': 1., 'right': 2.}
    assert rows[-1]['terminated'] and r['points_lost'] == 3


def test_horizon_not_reported_as_complete_and_no_overwrite(tmp_path):
    r = trace_episode(Agent(), Env(), 320101, 2, tmp_path / 'trace')
    assert r['stop_reason'] == 'horizon' and r['steps'] == 2
    with pytest.raises(FileExistsError):
        trace_episode(Agent(), Env(), 320101, 2, tmp_path / 'trace')


@pytest.mark.parametrize('learner', ['corrected_common', 'bart'])
@pytest.mark.parametrize('backend', ['legacy_visual_pong_v1', 'ocatari_ram_v1'])
@pytest.mark.atari
@pytest.mark.ocatari
def test_real_headless_frames_and_greedy_trace(tmp_path, learner, backend):
    from bayesian_rrtl.atari import AtariConfig
    from bayesian_rrtl.baseline_adapter import BaselineLearnerConfig
    from bayesian_rrtl.comparison import make_comparison_session
    from bayesian_rrtl.games.pong import PONG_ACTIONS
    from bayesian_rrtl.q_config import BayesianQConfig
    env = AtariConfig(game='Pong', extractor=backend, include_incomplete_states=backend == 'ocatari_ram_v1')
    config = BaselineLearnerConfig(seed=81) if learner == 'corrected_common' else BayesianQConfig(
        actions=PONG_ACTIONS, epsilon_decay_steps=50000, reward_min=env.reward_bounds[0],
        reward_max=env.reward_bounds[1], gamma=.99, seed=81, m=1, burn_in=1, draws=2,
        fit_interval=4, warmup_interactions=4, batch_size_per_action=4)
    session = make_comparison_session(learner, env, config)
    try:
        session.run(8)
        r = trace_episode(session.agent, session.environment, 320101, 5, tmp_path / 'trace', gif=True)
        assert r['steps'] == 5 and r['gif_frames'] == 2
        assert session.agent.interactions == 8
        assert (tmp_path / 'trace/preview.gif').exists()
    finally:
        session.environment.close()
