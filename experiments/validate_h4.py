"""Numerical acceptance evidence for reversible revisions and General BART."""

import argparse
from collections import namedtuple
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from bayesian_rrtl import RelationalEncoder
from bayesian_rrtl.ensemble import BARTRegressor
from bayesian_rrtl.exact import enumerate_trees, exact_posterior
from bayesian_rrtl.general_bart import GeneralBART, LinearMean
from bayesian_rrtl.priors import TreePrior
from bayesian_rrtl.proposals import RevisionKernel, log_acceptance_ratio
from bayesian_rrtl.rules import RuleCatalog


def validate(output):
    output.mkdir(parents=True, exist_ok=False)
    fact = namedtuple("Fact", "type value name obj1 obj2 obj3", defaults=[None, None])
    states = [
        [fact("logical", str(a), "a", "object"), fact("logical", str(b), "b", "object")]
        for a, b in [(False, False), (False, True), (True, False), (True, True)]
    ]
    encoder = RelationalEncoder.from_states(states)
    batch = encoder.transform(states)
    rules, prior = RuleCatalog(encoder), TreePrior(max_depth=2, alpha_tree=0.6)
    trees = enumerate_trees(batch, rules, prior)
    index = {t.structure_key(): i for i, t in enumerate(trees)}
    y = np.array([-0.2, 0.1, 0.2, -0.1])
    pi = exact_posterior(trees, batch, y, 0.1, 0.0625, prior, rules)
    kernel = RevisionKernel(batch, rules, prior)
    matrix = np.zeros((len(trees), len(trees)))
    nontrivial_swaps = 0
    for i, t in enumerate(trees):
        for p in kernel.all_proposals(t):
            probability = math.exp(p.log_q_forward)
            if p.impossible:
                matrix[i, i] += probability
                continue
            j = index[p.candidate.structure_key()]
            if p.move == "swap" and i != j:
                nontrivial_swaps += 1
            acceptance = math.exp(
                min(0, log_acceptance_ratio(t, p, batch, y, 0.1, 0.0625, prior, rules))
            )
            matrix[i, j] += probability * acceptance
            matrix[i, i] += probability * (1 - acceptance)
    flow = pi[:, None] * matrix
    balance = float(np.abs(flow - flow.T).max())
    normalization = float(np.abs(matrix.sum(axis=1) - 1).max())
    assert balance < 1e-13 and normalization < 1e-13 and nontrivial_swaps > 0
    model = BARTRegressor(m=2, kernel="revision")
    a = model.fit(batch, y, np.random.default_rng(17), burn_in=10, draws=30)
    b = GeneralBART(model).fit(
        batch, y, np.random.default_rng(17), burn_in=10, draws=30
    )
    assert a.draws == b.draws and a.trace == b.trace
    design = np.column_stack((np.ones(30), np.linspace(-1, 1, 30)))
    response = design @ [0.1, 0.3] + np.random.default_rng(2).normal(0, 0.1, 30)
    empty = RelationalEncoder([]).transform([[]] * 30)
    posterior = GeneralBART(additional=LinearMean(np.diag([0.4, 0.2]))).fit(
        empty,
        response,
        np.random.default_rng(91),
        design=design,
        trees_enabled=False,
        center_design=False,
        fixed_sigma2=0.01,
        burn_in=0,
        draws=12000,
    )
    precision = np.diag([1 / 0.4, 1 / 0.2]) + design.T @ design / 0.01
    covariance = np.linalg.solve(precision, np.eye(2))
    mean = np.linalg.solve(precision, design.T @ response / 0.01)
    mean_error = float(np.abs(posterior.theta.mean(axis=0) - mean).max())
    covariance_error = float(np.abs(np.cov(posterior.theta.T) - covariance).max())
    assert mean_error < 0.0007 and covariance_error < 0.000025
    report = {
        "milestone": "H4",
        "structures": len(trees),
        "nontrivial_swap_events": nontrivial_swaps,
        "max_detailed_balance_error": balance,
        "max_row_normalization_error": normalization,
        "zero_mean_draw_for_draw": True,
        "linear_draws": 12000,
        "linear_mean_error": mean_error,
        "linear_covariance_error": covariance_error,
        "limitations": [
            "Exact finite-state check fixes sigma2 and marginalizes leaves.",
            "General linear extension validated synthetically; no arbitrary noise models or OCAtari.",
        ],
    }
    root = Path(__file__).resolve().parents[1]
    report["source_hashes"] = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in (
            "bayesian_rrtl/general_bart.py",
            "bayesian_rrtl/proposals.py",
            "bayesian_rrtl/rules.py",
        )
    }
    (output / "report.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(validate(parser.parse_args().output), indent=2))
