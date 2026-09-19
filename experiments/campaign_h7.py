"""Predeclared multigame campaign, ablations, controls and separate test evaluation.

Development campaigns exercise the protocol with small budgets
they cannot be
reported as the full efficacy study. Test mode uses only saved final checkpoints.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import tempfile
import time

import numpy as np

from bayesian_rrtl.agent import BayesianQAgent
from bayesian_rrtl.atari import AtariConfig, AtariRelationalEnvironment, game_actions
from bayesian_rrtl.checkpoint import load_checkpoint
from bayesian_rrtl.controls import BatchTreeAgent, IncrementalBinaryAgent
from bayesian_rrtl.evaluation import (
    seed_summary,
    paired_bootstrap,
    learning_curve_area,
    export_rules,
)
from bayesian_rrtl.q_config import BayesianQConfig
from bayesian_rrtl.training import QTrainingSession

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = (
    "bart",
    "bart_revision",
    "single_tree",
    "batch_tree",
    "incremental_binary",
    "corrected",
)


def validate_protocol(p):
    if p["purpose"] not in ("development", "efficacy"):
        raise ValueError("Declare campaign purpose")
    if (
        not p["games"]
        or len(set(p["games"])) != len(p["games"])
        or set(p["games"]) - {"Breakout", "Pong", "DemonAttack"}
    ):
        raise ValueError("Invalid games")
    if (
        len(set(p["training_seeds"])) != len(p["training_seeds"])
        or len(p["training_seeds"]) < 3
    ):
        raise ValueError("Need >=3 distinct training seeds")
    if (
        not p["variants"]
        or len(set(p["variants"])) != len(p["variants"])
        or set(p["variants"]) - set(VARIANTS)
    ):
        raise ValueError("Invalid ablations")
    for name in ("interactions", "curve_interval", "max_episode_steps"):
        if type(p[name]) is not int or p[name] < 1:
            raise ValueError("Invalid campaign budget")
    seeds = [*p["training_seeds"], *p["validation_seeds"], *p["test_seeds"]]
    if (
        len(set(seeds)) != len(seeds)
        or not p["validation_seeds"]
        or not p["test_seeds"]
    ):
        raise ValueError("All training/validation/test seeds must be globally disjoint")
    if any(type(s) is not int or not 0 <= s < 2**32 for s in seeds):
        raise ValueError("Invalid seed")
    if set(p["failure_thresholds"]) != set(p["games"]):
        raise ValueError("Predeclare failure threshold for each game")
    if any(not np.isfinite(v) for v in p["failure_thresholds"].values()):
        raise ValueError("Failure thresholds must be finite")
    if any(
        len(pair) != 2 or any(v not in p["variants"] for v in pair)
        for pair in p["contrasts"]
    ):
        raise ValueError("All contrast variants must be predeclared")
    control = p.get("incremental_control", {"eta": 0.025, "split_min": 16})
    if "incremental_binary" in p["variants"] and (
        not 0 < control["eta"] <= 1
        or not 2 <= control["split_min"] <= p["agent"]["batch_size_per_action"]
    ):
        raise ValueError("Incremental split_min must fit in its leaf target buffer")
    for game in p["games"]:
        for seed in p["training_seeds"]:
            for variant in p["variants"]:
                settings(p, game, seed, variant)


def settings(protocol, game, seed, variant):
    environment = AtariConfig(
        game=game,
        extractor=f"legacy_visual_{game.lower()}_v1",
        representation=protocol.get("representation", "comparative"),
        include_incomplete_states=protocol.get("include_incomplete_states", False),
        max_episode_steps=protocol["max_episode_steps"],
    )
    options = dict(protocol["agent"])
    options.update(
        actions=game_actions(game),
        seed=seed,
        reward_min=-0.9 if game == "Pong" else -1.0,
        reward_max=1.1 if game == "Pong" else 1.0,
        validation_seeds=protocol["validation_seeds"],
        test_seeds=protocol["test_seeds"],
        resample_repeated_actions=game == "Breakout",
        kernel="revision" if variant == "bart_revision" else "grow_prune",
    )
    if variant == "single_tree":
        options["m"] = 1
    return environment, BayesianQConfig(**options)


def run_one(protocol, game, seed, variant, output):
    from train_atari import run_corrected, manifest

    env_config, config = settings(protocol, game, seed, variant)
    output.mkdir(parents=True, exist_ok=False)
    (output / "manifest.json").write_text(
        json.dumps(
            dict(
                manifest(env_config, config, variant),
                campaign_source_sha256=hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
                incremental_control=protocol.get(
                    "incremental_control", {"eta": 0.025, "split_min": 16}
                ),
            ),
            indent=2,
        )
    )
    if variant == "corrected":
        summary = run_corrected(
            env_config, config, output, protocol["interactions"], "validation"
        )
        # H0 final-only API does not expose intermediate policy evaluation.
        summary["curve"] = None
        (output / "summary.json").write_text(
            json.dumps(summary, indent=2, allow_nan=False)
        )
        return summary
    cls = {
        "batch_tree": BatchTreeAgent,
        "incremental_binary": IncrementalBinaryAgent,
    }.get(variant, BayesianQAgent)
    env = AtariRelationalEnvironment(env_config)
    control_options = (
        protocol.get("incremental_control", {"eta": 0.025, "split_min": 16})
        if cls is IncrementalBinaryAgent
        else {}
    )
    agent = cls(config, env.encoder(), **control_options)
    session = QTrainingSession(agent, env)
    curve = []
    training_seconds = 0.0

    def evaluate_point():
        rows = agent.evaluate(
            lambda: AtariRelationalEnvironment(env_config), config.validation_seeds
        )
        curve.append(
            {
                "interactions": agent.interactions,
                "training_seconds": training_seconds,
                "mean_return": float(np.mean([r["return"] for r in rows])),
            }
        )
        return rows

    try:
        evaluate_point()
        while agent.interactions < protocol["interactions"]:
            count = min(
                protocol["curve_interval"],
                protocol["interactions"] - agent.interactions,
            )
            start = time.perf_counter()
            session.run(count)
            training_seconds += time.perf_counter() - start
            rows = evaluate_point()
            print(
                json.dumps(
                    {
                        "game": game,
                        "seed": seed,
                        "variant": variant,
                        "interactions": agent.interactions,
                    }
                ),
                flush=True,
            )
        checkpoint = session.save(output / "final_checkpoint.rrtl")
        # Verify saved artifact before final evaluation/export.
        restored, _ = load_checkpoint(checkpoint)
        assert restored.interactions == protocol["interactions"]
        reference = agent.encoder.transform([])
        if agent.replay.transitions:
            from bayesian_rrtl.replay import concatenate_states

            reference = concatenate_states(
                [t.state for t in agent.replay.transitions[:8]], agent.encoder
            )
        predictions = agent.model.predict(reference)
        if not np.isfinite(predictions).all():
            raise ValueError("Non-finite predictions")
        bayesian = variant in ("bart", "bart_revision", "single_tree")
        rules = export_rules(agent.model, reference, bayesian=bayesian)
        complexities = []
        for posterior in agent.model.posteriors:
            if posterior is None:
                continue
            trees = posterior.draws[-1].trees if bayesian else (posterior.tree,)
            paths = [path for tree in trees for path in tree.partition(reference)]
            complexities.append(
                {"leaves": len(paths), "max_depth": max(map(len, paths))}
            )
        disagreement = None
        if bayesian and all(p is not None for p in agent.model.posteriors):
            samples = np.stack(
                [p.predict_latent_draws(reference) for p in agent.model.posteriors],
                axis=-1,
            )
            actions = samples.argmax(axis=-1)
            disagreement = float(
                np.mean(
                    [
                        1 - np.bincount(a, minlength=len(config.actions)).max() / len(a)
                        for a in actions.T
                    ]
                )
            )
        summary = {
            "variant": variant,
            "game": game,
            "seed": seed,
            "interactions": agent.interactions,
            "training_seconds": training_seconds,
            "evaluation": rows,
            "curve": curve,
            "validation_auc": learning_curve_area(curve),
            "complexity": complexities,
            "posterior_policy_disagreement": disagreement,
            "max_abs_q": float(np.abs(predictions).max()),
            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * (1 if sys.platform == "darwin" else 1024),
            "finite_data": all(
                np.isfinite(r["reward"]) and np.isfinite(r["raw_reward"])
                for r in session.history
            ),
            "fits": len(agent.fit_history),
            "model_id": agent.model.model_id,
            "training_seeds": agent.training_seeds,
        }
        for filename, data in [
            ("summary.json", summary),
            ("validation.json", rows),
            ("train.json", session.history),
            ("fits.json", agent.fit_history),
            ("rules.json", rules),
            ("encoder.json", agent.encoder.to_dict()),
        ]:
            (output / filename).write_text(json.dumps(data, indent=2, allow_nan=False))
        return summary
    finally:
        env.close()


def aggregate(protocol, output, split):
    report = {
        "purpose": protocol["purpose"],
        "split": split,
        "games": {},
        "protocol_hash": hashlib.sha256(
            json.dumps(protocol, sort_keys=True).encode()
        ).hexdigest(),
        "limitations": [
            "Development runs do not establish effectiveness or posterior convergence.",
            "Intervals resample training seeds, not individual evaluation episodes.",
            "No cross-game average of raw returns; all comparisons are exploratory.",
            "Corrected baseline has final evaluation only; no curve/AUC is imputed.",
            "General BART linear component is synthetic-only; no Atari covariate extension.",
            "Visual extraction only; OCAtari remains deferred.",
        ],
    }
    for game in protocol["games"]:
        variants = {}
        values = {}
        for variant in protocol["variants"]:
            returns = {}
            runs = []
            for seed in protocol["training_seeds"]:
                folder = output / game / variant / str(seed)
                summary = json.loads((folder / "summary.json").read_text())
                if (
                    summary["interactions"] != protocol["interactions"]
                    or not summary["finite_data"]
                ):
                    raise ValueError("Incomplete/nonfinite campaign run")
                rows = json.loads((folder / f"{split}.json").read_text())
                if split == "test":
                    validate_test_rows(rows, protocol["test_seeds"], folder / "test.json")
                if [r["seed"] for r in rows] != protocol[f"{split}_seeds"]:
                    raise ValueError("Evaluation seed mismatch")
                returns[seed] = float(np.mean([r["return"] for r in rows]))
                runs.append(
                    {
                        "seed": seed,
                        "return": returns[seed],
                        "training_seconds": summary["training_seconds"],
                        "validation_auc": summary.get("validation_auc"),
                        "complexity": summary.get("complexity"),
                        "posterior_policy_disagreement": summary.get(
                            "posterior_policy_disagreement"
                        ),
                    }
                )
            values[variant] = returns
            variants[variant] = dict(
                seed_summary(
                    list(returns.values()),
                    failure_threshold=protocol["failure_thresholds"][game],
                ),
                runs=runs,
            )
        contrasts = {}
        for left, right in protocol["contrasts"]:
            if left not in values or right not in values:
                raise ValueError("Contrast variant absent")
            contrasts[f"{left} - {right}"] = paired_bootstrap(
                values[left], values[right]
            )
        report["games"][game] = {"variants": variants, "paired_contrasts": contrasts}
    (output / f"report_{split}.json").write_text(
        json.dumps(report, indent=2, allow_nan=False)
    )
    lines = [
        f"# H7 — {protocol['purpose']} / {split}",
        "",
        "Comparaciones exploratorias por semilla de entrenamiento; sin conclusión confirmatoria.",
        "",
        "| Juego | Variante | Semillas | Media | Mediana | IQR |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for game, g in report["games"].items():
        for variant, v in g["variants"].items():
            lines.append(
                f"| {game} | {variant} | {v['n_training_seeds']} | {v['mean']:.3f} | {v['median']:.3f} | {v['q75'] - v['q25']:.3f} |"
            )
    (output / f"report_{split}.md").write_text("\n".join(lines) + "\n")
    return report


def validate_test_rows(rows, seeds, path):
    """Require one completed, finite episode for each reserved seed, in order."""
    valid = isinstance(rows, list) and len(rows) == len(seeds)
    if valid:
        for row, seed in zip(rows, seeds):
            if not isinstance(row, dict) or not (
                type(row.get("seed")) is int and row["seed"] == seed
                and type(row.get("return")) in (int, float)
                and np.isfinite(row["return"])
                and type(row.get("steps")) is int and row["steps"] > 0
                and type(row.get("terminated")) is bool
                and type(row.get("truncated")) is bool
                and (row["terminated"] or row["truncated"])
            ):
                valid = False
                break
    if not valid:
        raise ValueError(f"Invalid or incomplete test results: {path}")


def write_json_atomic(path, rows):
    """Publish only fully serialized results, on the destination filesystem."""
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(rows, stream, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def evaluate_test(protocol, output):
    """No training/tuning; evaluate every predeclared final checkpoint, never select a winner."""
    complete = set()
    for game in protocol["games"]:
        for variant in protocol["variants"]:
            for seed in protocol["training_seeds"]:
                folder = output / game / variant / str(seed)
                path = folder / "test.json"
                if path.exists():
                    try:
                        rows = json.loads(path.read_text())
                    except (ValueError, UnicodeError) as exc:
                        raise ValueError(f"Invalid test JSON: {path}") from exc
                    validate_test_rows(rows, protocol["test_seeds"], path)
                    complete.add(path)
    for game in protocol["games"]:
        for variant in protocol["variants"]:
            for seed in protocol["training_seeds"]:
                folder = output / game / variant / str(seed)
                path = folder / "test.json"
                if path in complete:
                    continue
                env_config, config = settings(protocol, game, seed, variant)
                if variant == "corrected":
                    from bayesian_rrtl.baseline import CorrectedRRLAgent
                    from bayesian_rrtl.config import BaselineConfig

                    c = BaselineConfig.load(folder / "baseline_config.json")
                    agent = CorrectedRRLAgent(c, folder)
                    agent.load_final_checkpoint(folder / "final_checkpoint.pkl")
                    rows = agent.evaluate(c.test_seeds)
                else:
                    agent, _ = load_checkpoint(
                        folder / "final_checkpoint.rrtl", expected_config=config
                    )
                    rows = agent.evaluate(
                        lambda: AtariRelationalEnvironment(env_config),
                        config.test_seeds,
                    )
                validate_test_rows(rows, protocol["test_seeds"], path)
                write_json_atomic(path, rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=("train", "test", "report-validation", "report-test"),
        default="train",
    )
    parser.add_argument("--worker", nargs=3, metavar=("GAME", "SEED", "VARIANT"))
    args = parser.parse_args()
    if args.worker:
        protocol = json.loads(args.protocol.read_text())
        validate_protocol(protocol)
        game, seed, variant = args.worker
        run_one(protocol, game, int(seed), variant, args.output)
        return
    if args.stage == "train":
        if args.protocol is None:
            parser.error("Training requires a predeclared --protocol")
        protocol = json.loads(args.protocol.read_text())
        validate_protocol(protocol)
        args.output.mkdir(parents=True, exist_ok=False)
        frozen = args.output.resolve() / "protocol.json"
        frozen.write_text(json.dumps(protocol, indent=2))
        (args.output / "protocol.sha256").write_text(
            hashlib.sha256(frozen.read_bytes()).hexdigest()
        )
        for game in protocol["games"]:
            for variant in protocol["variants"]:
                for seed in protocol["training_seeds"]:
                    subprocess.run(
                        [
                            sys.executable,
                            "-m",
                            "experiments.campaign_h7",
                            "--protocol",
                            str(frozen),
                            "--output",
                            str((args.output / game / variant / str(seed)).resolve()),
                            "--worker",
                            game,
                            str(seed),
                            variant,
                        ],
                        cwd=ROOT,
                        check=True,
                    )
        aggregate(protocol, args.output, "validation")
    else:
        if args.protocol is not None:
            parser.error("Use only the saved protocol after training")
        frozen = args.output / "protocol.json"
        if (
            hashlib.sha256(frozen.read_bytes()).hexdigest()
            != (args.output / "protocol.sha256").read_text()
        ):
            raise ValueError("Frozen protocol changed")
        protocol = json.loads(frozen.read_text())
        validate_protocol(protocol)
        if args.stage == "test":
            # First verify all runs are complete before touching reserved test seeds.
            aggregate(protocol, args.output, "validation")
            evaluate_test(protocol, args.output)
        aggregate(
            protocol,
            args.output,
            "test" if args.stage in ("test", "report-test") else "validation",
        )


if __name__ == "__main__":
    main()
