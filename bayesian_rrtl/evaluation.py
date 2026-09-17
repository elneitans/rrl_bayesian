"""Seed-level reports and paired bootstrap; episodes are not independent runs."""

import numpy as np


def seed_summary(values, *, failure_threshold):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Need finite returns per training seed")
    return {
        "n_training_seeds": len(values),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "std": float(values.std(ddof=1)) if len(values) > 1 else None,
        "q25": float(np.quantile(values, 0.25)),
        "q75": float(np.quantile(values, 0.75)),
        "failure_threshold": failure_threshold,
        "failure_fraction": float(np.mean(values < failure_threshold)),
    }


def paired_bootstrap(left, right, *, seed=912345, draws=5000):
    if set(left) != set(right) or not left:
        raise ValueError("Paired comparisons need the same nonempty training seeds")
    keys = sorted(left)
    differences = np.array([left[k] - right[k] for k in keys], dtype=float)
    if not np.isfinite(differences).all():
        raise ValueError("Non-finite differences")
    bootstrap = (
        np.random.default_rng(seed)
        .choice(differences, (draws, len(keys)), replace=True)
        .mean(axis=1)
    )
    return {
        "training_seeds": keys,
        "mean_difference": float(differences.mean()),
        "bootstrap_95_percentile": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
        "bootstrap_draws": draws,
        "bootstrap_seed": seed,
        "interpretation": "Exploratory seed-level interval; no multiplicity-adjusted confirmatory claim.",
    }


def learning_curve_area(points):
    points = sorted(points, key=lambda p: p["interactions"])
    x = np.array([p["interactions"] for p in points], dtype=float)
    y = np.array([p["mean_return"] for p in points], dtype=float)
    if len(x) < 2 or np.any(np.diff(x) <= 0) or not np.isfinite(y).all():
        raise ValueError("Learning curve needs distinct increasing checkpoints")
    return float(np.trapz(y, x) / (x[-1] - x[0]))


def export_rules(model, batch, *, bayesian=True):
    """Posterior inclusion frequency by action; no claim of one faithful program."""
    from dataclasses import asdict
    import json

    output = {}
    for action, posterior in zip(model.actions, model.posteriors):
        if posterior is None:
            output[action] = {"fitted": False}
            continue
        draws = [d.trees for d in posterior.draws] if bayesian else [(posterior.tree,)]
        counts = {}
        contributions = []
        for trees in draws:
            seen = set()
            for tree in trees:
                for _, node, _ in tree.walk(batch):
                    if not node.is_leaf:
                        seen.add(json.dumps(asdict(node.rule), sort_keys=True))
            for rule in seen:
                counts[rule] = counts.get(rule, 0) + 1
        trees = draws[-1]
        for index, tree in enumerate(trees):
            contributions.append(
                {"tree": index, "reference_contribution": tree.predict(batch).tolist()}
            )
        output[action] = {
            "fitted": True,
            "frequency_kind": "posterior inclusion"
            if bayesian
            else "point model inclusion",
            "rules": [
                {"rule": json.loads(rule), "frequency": count / len(draws)}
                for rule, count in sorted(counts.items())
            ],
            "last_draw_contributions": contributions,
            "units": "normalized; add intercept and multiply scale once per ensemble"
            if bayesian
            else "original reward units",
        }
    return output
