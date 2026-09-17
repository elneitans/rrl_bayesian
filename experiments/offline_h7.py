"""Regressor/chain-length ablation on identical saved transitions and TD targets.

Errors are in-sample fit diagnostics, not held-out RL effectiveness estimates.
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np

from bayesian_rrtl.checkpoint import load_checkpoint
from bayesian_rrtl.controls import CARTRegressor
from bayesian_rrtl.diagnostics import posterior_diagnostics
from bayesian_rrtl.ensemble import BARTRegressor


def run(checkpoint, output, *, burn_in=50, draws=50, chains=4, seed=700001):
    agent, _ = load_checkpoint(checkpoint)
    if agent.last_fit is None or chains < 2 or draws < 4:
        raise ValueError("Need completed fit, >=2 chains and >=4 draws")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    c = agent.config
    report = {
        "checkpoint": str(checkpoint),
        "fit_id": agent.last_fit.fit_id,
        "config_hash": c.config_hash,
        "burn_in": burn_in,
        "draws": draws,
        "chains": chains,
        "seed": seed,
        "actions": {},
        "interpretation": "Fixed training dataset and targets; MSE is in-sample, not a validation score.",
    }
    for dataset in agent.last_fit.datasets:
        if not len(dataset.batch):
            continue
        name = c.actions[dataset.action]
        results = {}
        cart = CARTRegressor(c).fit(dataset.batch, dataset.targets, None)
        results["batch_tree"] = {
            "train_mse": float(
                np.mean((cart.predict_mean(dataset.batch) - dataset.targets) ** 2)
            ),
            "seconds": cart.elapsed_seconds,
        }
        for label, m, kernel, multiplier in [
            ("single_tree", 1, "grow_prune", 1),
            ("bart", c.m, "grow_prune", 1),
            ("bart_revision", c.m, "revision", 1),
            ("bart_longer", c.m, "grow_prune", 2),
        ]:
            streams = np.random.SeedSequence([seed, dataset.action]).spawn(chains)
            posteriors = []
            started = time.perf_counter()
            for stream in streams:
                posterior = BARTRegressor(
                    m=m,
                    kernel=kernel,
                    tree_prior=c.tree_prior,
                    scale=c.scale,
                    k=c.k,
                    noise_prior=agent.calibrations[dataset.action].prior,
                ).fit(
                    dataset.batch,
                    dataset.targets,
                    np.random.default_rng(stream),
                    burn_in=burn_in * multiplier,
                    draws=draws * multiplier,
                )
                posteriors.append(posterior)
            reference = dataset.batch.take(
                np.unique(dataset.batch.values, axis=0, return_index=True)[1][:4]
            )
            diagnostics = posterior_diagnostics(posteriors, reference)
            predictions = np.mean(
                [p.predict_mean(dataset.batch) for p in posteriors], axis=0
            )
            results[label] = {
                "train_mse": float(np.mean((predictions - dataset.targets) ** 2)),
                "seconds": time.perf_counter() - started,
                "diagnostics": diagnostics,
                "all_meet_rhat_1_01_ess_400": all(
                    d["status"] == "ok"
                    and d["rhat"] <= 1.01
                    and d["ess_bulk"] >= 400
                    and d["ess_tail"] >= 400
                    for d in diagnostics.values()
                ),
            }
            print(
                json.dumps(
                    {
                        "action": name,
                        "variant": label,
                        "seconds": results[label]["seconds"],
                    }
                ),
                flush=True,
            )
        report["actions"][name] = {
            "transition_ids": dataset.transition_ids,
            "dataset_hash": dataset.data_hash,
            "samples": len(dataset.batch),
            "variants": results,
        }
        (output / "report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False)
        )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--burn-in", type=int, default=50)
    parser.add_argument("--draws", type=int, default=50)
    parser.add_argument("--chains", type=int, default=4)
    args = parser.parse_args()
    run(
        args.checkpoint,
        args.output,
        burn_in=args.burn_in,
        draws=args.draws,
        chains=args.chains,
    )
