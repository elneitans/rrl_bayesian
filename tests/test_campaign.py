from dataclasses import replace

import numpy as np
import pytest

from bayesian_rrtl.agent import BayesianQAgent
from bayesian_rrtl.atari import AtariConfig, AtariRelationalEnvironment, game_actions
from bayesian_rrtl.checkpoint import load_checkpoint, save_checkpoint
from bayesian_rrtl.controls import BatchTreeAgent, IncrementalBinaryAgent
from bayesian_rrtl.q_config import BayesianQConfig
from bayesian_rrtl.toy_mdp import TwoStepMDP
from bayesian_rrtl.training import QTrainingSession


@pytest.mark.atari
@pytest.mark.parametrize("game", ["Pong", "DemonAttack"])
@pytest.mark.parametrize("representation", ["comparative", "logical"])
def test_multigame_objects_match_historical_facts_and_resume(game, representation):
    import preprocessing

    config = AtariConfig(
        game=game,
        extractor=f"legacy_visual_{game.lower()}_v1",
        representation=representation,
        include_incomplete_states=True,
        max_episode_steps=22,
    )
    a, b = AtariRelationalEnvironment(config), AtariRelationalEnvironment(config)
    legacy = getattr(preprocessing, game + "Preprocessor")()
    suffix = "pong" if game == "Pong" else "demon_attack"
    builder = getattr(preprocessing, f"get_state_{representation}_{suffix}")
    previous = None
    try:
        obs, _ = a.env.reset(seed=34)
        for _ in range(16):
            state = a._state(obs)
            info = legacy.get_info(obs)
            assert state == builder(info, previous, True)
            a.encoder().transform([state])
            previous = info
            obs, _, term, trunc, _ = a.env.step(1)
            if term or trunc:
                break
        a.done = False
        b.restore(a.snapshot())
        for action in (1, 2, 3):
            x, y = a.step(action), b.step(action)
            assert x == y
            assert x[1] == np.sign(x[2]) + (0.1 if game == "Pong" else 0)
            if x[-1] or x[-2]:
                break
        assert a.actions == game_actions(game)
    finally:
        a.close()
        b.close()


def test_batch_control_has_same_frozen_targets_and_ids_as_bart():
    c = BayesianQConfig(m=1, burn_in=2, draws=4, fit_interval=8)
    agents = [cls(c, TwoStepMDP.encoder()) for cls in (BayesianQAgent, BatchTreeAgent)]
    for a in agents:
        for i in range(8):
            a.observe(
                TwoStepMDP.state(i % 2),
                i % 2,
                float(i % 2),
                None,
                True,
                False,
                raw_reward=float(i % 2),
                episode_id=i,
                step_id=0,
            )
        a.fit_if_due()
    for x, y in zip(agents[0].last_fit.datasets, agents[1].last_fit.datasets):
        assert x.transition_ids == y.transition_ids and x.data_hash == y.data_hash
        np.testing.assert_array_equal(x.targets, y.targets)


@pytest.mark.parametrize("cls", [BatchTreeAgent, IncrementalBinaryAgent])
def test_control_resume_and_known_q(cls, tmp_path):
    config = BayesianQConfig(
        seed=11, fit_interval=100, batch_size_per_action=256, epsilon_decay_steps=2000
    )
    kwargs = {"eta": 0.2, "split_min": 8} if cls is IncrementalBinaryAgent else {}
    agent = cls(config, TwoStepMDP.encoder(), **kwargs)
    session = QTrainingSession(agent, TwoStepMDP())
    session.run(301)
    path = session.save(tmp_path / "model.rrtl")
    restored = QTrainingSession.load(path, TwoStepMDP())
    session.run(299)
    restored.run(299)
    assert session.history == restored.history
    batch = agent.encoder.transform([TwoStepMDP.state(0), TwoStepMDP.state(1)])
    np.testing.assert_array_equal(
        agent.model.predict(batch), restored.agent.model.predict(batch)
    )
    np.testing.assert_allclose(
        agent.model.predict(batch), TwoStepMDP.optimal_q(0.8), atol=0.1
    )
    with pytest.raises(ValueError, match="posterior"):
        agent.model.posteriors[0].predict_latent_draws(batch)


def test_revision_config_checkpoint_roundtrip(tmp_path):
    c = BayesianQConfig(kernel="revision", m=1, burn_in=1, draws=2, fit_interval=4)
    a = BayesianQAgent(c, TwoStepMDP.encoder())
    QTrainingSession(a, TwoStepMDP()).run(4)
    b, _ = load_checkpoint(save_checkpoint(a, tmp_path / "revision.rrtl"))
    assert b.config.kernel == "revision"
    with pytest.raises(ValueError):
        replace(c, kernel="invalid")


def test_seed_level_statistics_and_paired_order():
    from bayesian_rrtl.evaluation import (
        paired_bootstrap,
        seed_summary,
        learning_curve_area,
    )

    report = paired_bootstrap({3: 4.0, 1: 2.0, 2: 3.0}, {2: 2.0, 3: 3.0, 1: 1.0})
    assert report["mean_difference"] == 1 and report["bootstrap_95_percentile"] == [
        1.0,
        1.0,
    ]
    with pytest.raises(ValueError):
        paired_bootstrap({1: 2.0}, {2: 2.0})
    summary = seed_summary([0.0, 1.0, 5.0], failure_threshold=1.0)
    assert summary["failure_fraction"] == 1 / 3 and summary["n_training_seeds"] == 3
    assert (
        learning_curve_area(
            [
                {"interactions": 0, "mean_return": 1.0},
                {"interactions": 10, "mean_return": 3.0},
            ]
        )
        == 2.0
    )


def test_campaign_protocol_rejects_leakage_before_training():
    import json
    from pathlib import Path
    from experiments.campaign_h7 import validate_protocol

    p = json.loads(
        (
            Path(__file__).resolve().parents[1] / "configs/campaign_h7_development.json"
        ).read_text()
    )
    validate_protocol(p)
    p["test_seeds"] = [p["training_seeds"][0]]
    with pytest.raises(ValueError, match="disjoint"):
        validate_protocol(p)


def test_rule_frequency_uses_draw_inclusion_not_tree_counts():
    from bayesian_rrtl.evaluation import export_rules
    from bayesian_rrtl.ensemble import BARTRegressor, EnsembleDraw
    from bayesian_rrtl.agent import BayesianQModel
    from bayesian_rrtl import Tree
    from bayesian_rrtl.rules import RuleCatalog

    encoder = TwoStepMDP.encoder()
    batch = encoder.transform([TwoStepMDP.state(0), TwoStepMDP.state(1)])
    rule = RuleCatalog(encoder).rules[0]
    tree = Tree(rule=rule, true_child=Tree(0.1), false_child=Tree(-0.1))
    p = BARTRegressor(m=2).fit(
        batch, [0, 1], np.random.default_rng(4), burn_in=0, draws=2
    )
    p = replace(
        p, draws=(EnsembleDraw((tree, tree), 0.1), EnsembleDraw((Tree(), Tree()), 0.1))
    )
    report = export_rules(BayesianQModel(("x",), (p,), encoder.catalog_hash), batch)
    assert report["x"]["rules"][0]["frequency"] == 0.5


@pytest.mark.atari
def test_campaign_cli_frozen_protocol_and_separate_test(tmp_path):
    import json
    from pathlib import Path
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[1]
    p = json.loads((root / "configs/campaign_h7_development.json").read_text())
    p.update(
        games=["Breakout"],
        variants=["batch_tree"],
        interactions=8,
        curve_interval=4,
        max_episode_steps=8,
        failure_thresholds={"Breakout": 1},
        contrasts=[],
    )
    p["agent"].update(fit_interval=4, warmup_interactions=0, batch_size_per_action=8)
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(p))
    output = tmp_path / "campaign"

    def run(*args):
        return subprocess.run(
            [sys.executable, "-m", "experiments.campaign_h7", *map(str, args)],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=60,
        )

    result = run("--protocol", path, "--output", output)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (output / "report_validation.json").exists()
    assert not list(output.glob("*/*/*/test.json"))
    checkpoints = {
        p: p.read_bytes() for p in output.glob("*/*/*/final_checkpoint.rrtl")
    }
    result = run("--output", output, "--stage", "test")
    assert result.returncode == 0, result.stdout + result.stderr
    assert len(list(output.glob("*/*/*/test.json"))) == 3
    assert all(p.read_bytes() == data for p, data in checkpoints.items())
    result = run("--output", output, "--stage", "test")
    assert result.returncode != 0 and "already exist" in result.stderr
    frozen = output / "protocol.json"
    frozen.write_text(frozen.read_text() + "\n")
    result = run("--output", output, "--stage", "report-test")
    assert result.returncode != 0 and "protocol changed" in result.stderr
