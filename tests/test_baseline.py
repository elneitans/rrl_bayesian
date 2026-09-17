import csv
from dataclasses import replace
import json
import pickle

import gymnasium as gym
import pytest

from bayesian_rrtl.baseline import CorrectedNode, CorrectedRRLAgent, FireReset, RecordReward
from bayesian_rrtl.config import BaselineConfig
from bayesian_rrtl.targets import bootstrap_mask, q_learning_update
from preprocessing import Fact
from relational_regresion_tree import Literal, Node


@pytest.mark.parametrize("terminated,truncated,bootstrap,expected", [
    (False, False, True, 5), (True, False, True, 2),
    (False, True, True, 5), (False, True, False, 2), (True, True, True, 2)])
def test_td_targets(terminated, truncated, bootstrap, expected):
    calls = []
    def next_value():
        calls.append(True)
        return 6
    result = q_learning_update(4, 2, next_value, eta_q=0.25, gamma=0.5,
                               terminated=terminated, truncated=truncated,
                               bootstrap_on_truncation=bootstrap)
    assert result.target == expected
    assert result.new_q == 4 + 0.25 * (expected - 4)
    assert len(calls) == int(result.bootstrap)


def test_truncation_requires_final_observation():
    with pytest.raises(ValueError, match="final observation"):
        bootstrap_mask(False, True, final_observation_available=False)
    assert bootstrap_mask(True, True, final_observation_available=False) == 0


def test_actual_agent_uses_next_state_and_executed_action(tmp_path):
    agent = CorrectedRRLAgent(BaselineConfig(eta_q=0.25, gamma=0.5), tmp_path)
    literal = Literal("x", "comparative", "player_t", "ball_t")
    for i, root in enumerate(agent.root_nodes.values()):
        root.literal = literal
        root.less_branch = Node(root.root_action)
        root.more_branch = Node(root.root_action)
        root.same_branch = Node(root.root_action)
        root.less_branch.q_value = [4, 10, 1, 0][i]
        root.more_branch.q_value = [0, 1, 6, 2][i]
    state = {Fact("comparative", "less", "x", "player_t", "ball_t")}
    next_state = {Fact("comparative", "more", "x", "player_t", "ball_t")}
    update = agent.update_transition(state, 0, 2, next_state, False, False)
    assert update.old_q == 4
    assert update.target == 5
    assert agent.root_nodes["NOOP"].less_branch.q_value == 4.25
    assert agent.root_nodes["FIRE"].less_branch.q_value == 10


class FakeEnv(gym.Env):
    action_space = gym.spaces.Discrete(4)
    observation_space = gym.spaces.Discrete(100)

    def __init__(self, length=2, terminal=True):
        self.length = length
        self.terminal = terminal
        self.closed = False
        self.steps = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        return 0, {"reset": True}

    def step(self, action):
        self.steps += 1
        done = self.steps == self.length
        return self.steps, 4.0, done and self.terminal, done and not self.terminal, {}

    def get_action_meanings(self):
        return ["NOOP", "FIRE", "RIGHT", "LEFT"]

    def close(self):
        self.closed = True


class FakePreprocessor:
    def get_info(self, observation):
        return observation


def empty_state(current, previous, incomplete):
    return set()


class FakeAgent(CorrectedRRLAgent):
    def __init__(self, config, output_dir, length=2, terminal=True):
        super().__init__(config, output_dir)
        self.preprocessor = FakePreprocessor()
        self.get_state = empty_state
        self.fake_length = length
        self.fake_terminal = terminal

    def make_train_environment(self):
        self.last_env = RecordReward(FakeEnv(self.fake_length, self.fake_terminal), self.game_name)
        return self.last_env

    def make_test_environment(self):
        return FakeEnv(self.fake_length, self.fake_terminal)


@pytest.mark.parametrize("length,terminal,completed", [(2, True, 1), (2, False, 1), (10, True, 0)])
def test_final_transition_return_and_checkpoint(tmp_path, length, terminal, completed):
    config = BaselineConfig(n_iterations=2)
    agent = FakeAgent(config, tmp_path, length, terminal)
    summary = agent.train()
    rows = list(csv.DictReader((tmp_path / "train.csv").open()))
    assert len(rows) == 2
    assert rows[-1]["Iteration"] == "2"
    assert rows[-1]["Budget exhausted"] == "True"
    assert float(rows[-1]["Episode return"]) == 2
    assert float(rows[-1]["Raw episode return"]) == 8
    assert summary["completed_episodes"] == completed
    assert summary["last_episode"]["complete"] == bool(completed)
    assert float(rows[-1]["Bootstrap"]) == (0 if terminal and completed else 1)
    assert agent.epsilon == pytest.approx(config.epsilon_init * agent.epsilon_decay_rate**2)
    restored = FakeAgent(config, tmp_path / "restored", length, terminal)
    restored.load_final_checkpoint(tmp_path / "final_checkpoint.pkl")
    assert restored.i_step == 2
    assert restored.rng.bit_generator.state == agent.rng.bit_generator.state
    assert [n.q_value for n in restored.root_nodes.values()] == [n.q_value for n in agent.root_nodes.values()]
    assert agent.last_env.unwrapped.closed


def test_evaluation_does_not_mutate_training_and_is_repeatable(tmp_path):
    agent = FakeAgent(BaselineConfig(n_iterations=3), tmp_path)
    agent.train()
    before = pickle.dumps((agent.root_nodes, agent.rng.bit_generator.state,
                           agent.environment_rng.bit_generator.state, agent.action_buffer, agent.epsilon))
    first = agent.evaluate([100001, 100002])
    second = agent.evaluate([100001, 100002])
    assert first == second
    assert [r["return"] for r in first] == [8, 8]
    after = pickle.dumps((agent.root_nodes, agent.rng.bit_generator.state,
                          agent.environment_rng.bit_generator.state, agent.action_buffer, agent.epsilon))
    assert before == after
    with pytest.raises(ValueError, match="overlap"):
        agent.evaluate(agent.training_seeds)


def test_repeated_seed_training_reproduces_log(tmp_path):
    config = BaselineConfig(n_iterations=7)
    for name in ("first", "second"):
        FakeAgent(config, tmp_path / name).train()
    assert (tmp_path / "first/train.csv").read_bytes() == (tmp_path / "second/train.csv").read_bytes()


def test_fire_reset_returns_current_observation_after_second_startup_termination():
    env = FireReset(FakeEnv(length=2))
    observation, info = env.reset(seed=7, options={})
    assert observation == 0
    assert info == {"reset": True}
    assert env.unwrapped.steps == 0


def test_config_round_trip_and_disjoint_seeds(tmp_path):
    config = BaselineConfig()
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config.to_dict()))
    assert BaselineConfig.load(path).config_hash == config.config_hash
    with pytest.raises(ValueError, match="disjoint"):
        replace(config, test_seeds=config.validation_seeds)
    with pytest.raises(ValueError):
        replace(config, n_iterations=0)


@pytest.mark.parametrize('criteria', ['p-value', 'f-ratio', 'max-q-value'])
@pytest.mark.parametrize('model_version', ['comparative', 'logical'])
def test_constant_targets_skip_split_then_recover(tmp_path, criteria, model_version):
    config = BaselineConfig(min_sample_size=1, best_literal_criteria=criteria,
                            model_version=model_version, eta_q=1, significance_level=0.99)
    agent = CorrectedRRLAgent(config, tmp_path)
    values = ['less', 'more'] if model_version == 'comparative' else ['False', 'True']
    literal = agent.root_nodes['NOOP'].refinements[0]
    states = [{Fact(literal.type, value, literal.name, literal.obj1, literal.obj2)} for value in values]
    root = agent.root_nodes['NOOP']
    original_refinements = list(root.refinements)
    for _ in range(3):
        for state in states:
            agent.update_transition(state, 0, 0, state, True, False)
    assert root.literal is None
    assert root.refinements == original_refinements
    assert agent.n_splits == 0
    # Once targets vary, a previously deferred candidate must be usable again.
    agent.update_transition(states[1], 0, 1, states[1], True, False)
    assert root.literal is not None
    assert agent.n_splits == 1
    # Children are made by the legacy factory; protect their first split too.
    agent.update_transition(states[0], 0, 0, states[0], True, False)
    assert isinstance(root.get_state_leaf(states[0]), CorrectedNode)


def test_corrected_selection_skips_only_degenerate_candidate():
    literals = [Literal('x', 'comparative', 'player_t', 'ball_t'),
                Literal('y', 'comparative', 'player_t', 'ball_t')]
    node = CorrectedNode('NOOP', refinements=literals, min_sample_size=1)
    for literal in literals:
        node.refinements_stats[literal]['Total']['n'] = 3
    node.refinements_stats[literals[1]]['Total']['J'] = 2
    assert node.find_best_literal() == literals[1]
    assert node.refinements == literals


@pytest.mark.parametrize('extra', [['--eval-split', 'none'], ['--run', '2'],
                                   ['--results-dir', 'other'], ['--config', 'config.json']])
def test_evaluation_cli_rejects_training_overrides(monkeypatch, extra):
    import sys
    import train_baseline
    monkeypatch.setattr(sys, 'argv', ['train_baseline.py', '--checkpoint', 'missing.pkl', *extra])
    with pytest.raises(SystemExit) as error:
        train_baseline.main()
    assert error.value.code == 2


def test_evaluation_cli_reuses_checkpoint_config_without_training(tmp_path, monkeypatch):
    import sys
    import train_baseline
    import bayesian_rrtl.baseline as baseline
    config = BaselineConfig(n_iterations=2, run=7, test_seeds=(300001,))
    FakeAgent(config, tmp_path).train()
    checkpoint = tmp_path / 'final_checkpoint.pkl'
    saved = checkpoint.read_bytes()
    def no_training(self):
        pytest.fail('Evaluation must not train')
    monkeypatch.setattr(FakeAgent, 'train', no_training)
    monkeypatch.setattr(baseline, 'CorrectedRRLAgent', FakeAgent)
    monkeypatch.setattr(sys, 'argv', ['train_baseline.py', '--checkpoint', str(checkpoint),
                                     '--eval-split', 'test'])
    train_baseline.main()
    first = (tmp_path / 'test.json').read_bytes()
    assert json.loads(first)[0]['seed'] == 300001
    train_baseline.main()
    assert (tmp_path / 'test.json').read_bytes() == first
    assert checkpoint.read_bytes() == saved
