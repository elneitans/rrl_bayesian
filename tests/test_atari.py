import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import gymnasium as gym
import numpy as np
import pytest

from bayesian_rrtl import RelationalEncoder, RuleCatalog, Tree
from bayesian_rrtl.baseline import CorrectedRRLAgent
from bayesian_rrtl.config import BaselineConfig
from relational_regresion_tree import RRLAgent


@pytest.mark.atari
@pytest.mark.parametrize("game", ["Breakout", "Pong", "DemonAttack"])
@pytest.mark.parametrize("model_version", ["comparative", "logical"])
def test_real_atari_preprocessing_encoder_and_training(tmp_path, game, model_version):
    config = BaselineConfig(game=game, model_version=model_version, n_iterations=16,
                            max_episode_steps=10, validation_seeds=[100001], test_seeds=[200001])
    agent = CorrectedRRLAgent(config, tmp_path)
    summary = agent.train()
    assert summary["iterations"] == 16
    rows = list(csv.DictReader((tmp_path / "train.csv").open()))
    assert len(rows) == 16
    assert np.isfinite([float(row["New Q"]) for row in rows]).all()
    assert agent.evaluate(config.validation_seeds) == agent.evaluate(config.validation_seeds)
    # Exercise real Fact output, including partially observed states, through H1.
    env = agent.make_test_environment()
    try:
        observation, _ = env.reset(seed=12345)
        previous = None
        states = [set()]
        for _ in range(6):
            current = agent.preprocessor.get_info(observation)
            states.append(agent.get_state(current, previous, True))
            previous = current
            observation, _, terminated, truncated, _ = env.step(1)
            if terminated or truncated:
                break
        encoder = RelationalEncoder.from_states(states)
        restored = RelationalEncoder.from_dict(encoder.to_dict())
        batch = restored.transform(states)
        catalog = RuleCatalog(restored)
        groups = catalog.eligible(batch)
        assert groups, "At least observed/missing relations should separate real states"
        rule = next(iter(groups.values()))[0]
        tree = Tree(rule=rule, true_child=Tree(1), false_child=Tree(-1))
        tree.validate_support(batch, catalog)
        assert tree.predict(batch).shape == (len(states),)
        assert sum(len(indices) for leaf, indices in tree.partition(batch).values()) == len(states)
    finally:
        env.close()


@pytest.mark.atari
def test_historical_training_still_runs_unchanged(tmp_path):
    (tmp_path / "train/run_1").mkdir(parents=True)
    legacy = RRLAgent(env_name="BreakoutNoFrameskip-v4", game_name="Breakout", run=1,
                      save_dir=str(tmp_path) + "/", initial_seed=1)
    original_factory = legacy.make_train_environment
    def limited_environment():
        env = original_factory()
        # Put TimeLimit below the historical wrapper, whose reset lacks options.
        env.env = gym.wrappers.TimeLimit(env.env, max_episode_steps=10)
        return env
    legacy.make_train_environment = limited_environment
    legacy.relational_q_learning(n_iterations=17)
    assert list((tmp_path / "train/run_1").glob("*trees.pkl"))
    rows = list(csv.DictReader((tmp_path / "train/Breakout_run_1_train.csv").open()))
    # Characterization of the preserved legacy bug: final interaction is omitted.
    assert len(rows) == 16


@pytest.mark.atari
def test_runner_manifest_checkpoint_graph_and_reserved_test_seeds(tmp_path):
    config = BaselineConfig(n_iterations=16, max_episode_steps=10,
                            results_dir=str(tmp_path / "results"), render_graph=True)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config.to_dict()))
    root = Path(__file__).resolve().parents[1]
    # Running outside the repo must still resolve imports and paths correctly.
    result = subprocess.run([sys.executable, str(root / "train_baseline.py"), "--config", str(path)],
                            cwd=tmp_path, capture_output=True, text=True, env=os.environ, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    output = tmp_path / "results/rrtl_corrected/Breakout/comparative_version/run_1"
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["checkpoint_selection"] == "final"
    assert "bayesian_rrtl/baseline.py" in manifest["source_hashes"]
    assert (output / "final_checkpoint.pkl").is_file()
    assert (output / "final_tree.pdf").is_file()
    assert (output / "validation.json").is_file()
    assert not (output / "test.json").exists()
    preserved = {name: (output / name).read_bytes() for name in
                 ('train.csv', 'manifest.json', 'final_checkpoint.pkl', 'validation.json')}
    result = subprocess.run([sys.executable, str(root / 'train_baseline.py'),
                             '--checkpoint', str(output / 'final_checkpoint.pkl'),
                             '--eval-split', 'test'], cwd=tmp_path, capture_output=True,
                            text=True, env=os.environ, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    rows = json.loads((output / 'test.json').read_text())
    assert [row['seed'] for row in rows] == list(config.test_seeds)
    for name, data in preserved.items():
        assert (output / name).read_bytes() == data
