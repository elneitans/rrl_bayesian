import math

import numpy as np
import pytest

from bayesian_rrtl import RelationalEncoder
from bayesian_rrtl.ensemble import BARTRegressor
from bayesian_rrtl.exact import enumerate_trees, exact_posterior
from bayesian_rrtl.general_bart import GeneralBART, LinearMean
from bayesian_rrtl.priors import TreePrior
from bayesian_rrtl.proposals import RevisionKernel, log_acceptance_ratio
from test_bayesian_kernel import tiny_data


@pytest.mark.parametrize("depth,minimum", [(0, 1), (1, 1), (2, 1), (2, 2)])
def test_revision_kernel_exact_balance(depth, minimum):
    batch, y, rules = tiny_data()
    prior = TreePrior(alpha_tree=0.6, max_depth=depth, min_child=minimum)
    trees = enumerate_trees(batch, rules, prior)
    index = {tree.structure_key(): i for i, tree in enumerate(trees)}
    kernel = RevisionKernel(batch, rules, prior)
    matrix = np.zeros((len(trees), len(trees)))
    for i, tree in enumerate(trees):
        before = tree.structure_key()
        proposals = list(kernel.all_proposals(tree))
        assert sum(math.exp(p.log_q_forward) for p in proposals) == pytest.approx(1)
        for p in proposals:
            q = math.exp(p.log_q_forward)
            if p.impossible:
                matrix[i, i] += q
            else:
                j = index[p.candidate.structure_key()]
                accept = math.exp(
                    min(
                        0,
                        log_acceptance_ratio(
                            tree, p, batch, y, 0.1, 0.0625, prior, rules
                        ),
                    )
                )
                matrix[i, j] += q * accept
                matrix[i, i] += q * (1 - accept)
        assert tree.structure_key() == before
    posterior = exact_posterior(trees, batch, y, 0.1, 0.0625, prior, rules)
    np.testing.assert_allclose(matrix.sum(axis=1), 1, atol=1e-14)
    flow = posterior[:, None] * matrix
    np.testing.assert_allclose(flow, flow.T, atol=1e-14)
    np.testing.assert_allclose(posterior @ matrix, posterior, atol=1e-14)


def test_zero_mean_reproduces_h3_draw_for_draw():
    batch, y, _ = tiny_data()
    model = BARTRegressor(m=2, kernel="revision")
    a = model.fit(batch, y, np.random.default_rng(17), burn_in=10, draws=20)
    b = GeneralBART(model).fit(
        batch, y, np.random.default_rng(17), burn_in=10, draws=20
    )
    assert a.draws == b.draws and a.trace == b.trace
    assert {"change", "swap"} <= a.diagnostics()["moves"].keys()


def test_linear_only_recovers_analytic_posterior():
    design = np.column_stack((np.ones(30), np.linspace(-1, 1, 30)))
    y = design @ [0.1, 0.3] + np.random.default_rng(2).normal(0, 0.1, 30)
    batch = RelationalEncoder([]).transform([[]] * len(y))
    linear = LinearMean(np.diag([0.4, 0.2]))
    posterior = GeneralBART(additional=linear).fit(
        batch,
        y,
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
    mean = np.linalg.solve(precision, design.T @ y / 0.01)
    np.testing.assert_allclose(posterior.theta.mean(axis=0), mean, atol=0.0007)
    np.testing.assert_allclose(np.cov(posterior.theta.T), covariance, atol=0.000025)
    assert all(not trees for trees in posterior.trees)


def test_additive_signal_and_centering_prediction_contract():
    from test_bayesian_kernel import Fact

    rng = np.random.default_rng(42)
    x = np.tile([False, True], 50)
    w = rng.normal(3, 1, (100, 1))
    states = [[Fact("logical", str(bool(v)), "present", "ball")] for v in x]
    encoder = RelationalEncoder.from_states(states)
    batch = encoder.transform(states)
    signal = np.where(x, 0.2, -0.2) + 0.25 * (w[:, 0] - w.mean())
    y = signal + rng.normal(0, 0.025, len(signal))
    posterior = GeneralBART(
        BARTRegressor(m=2, tree_prior=TreePrior(max_depth=1)), LinearMean([[1.0]])
    ).fit(batch, y, rng, design=w, burn_in=150, draws=300)
    assert abs(posterior.theta.mean() - 0.25) < 0.03
    assert np.sqrt(np.mean((posterior.predict_mean(batch, w) - signal) ** 2)) < 0.04
    np.testing.assert_allclose(posterior.center, w.mean(axis=0))
    assert np.all(posterior.sigma2 > 0)
    assert not posterior.theta.flags.writeable
    with pytest.raises(ValueError):
        posterior.predict_mean(batch, w[:, 0])


def test_revision_contains_valid_swap_and_rejects_invalid_descendants():
    from test_bayesian_kernel import Fact

    states = [
        [Fact("logical", str(a), "a", "o"), Fact("logical", str(b), "b", "o")]
        for a, b in [(False, False), (False, True), (True, False), (True, True)]
    ]
    from bayesian_rrtl.rules import RuleCatalog

    encoder = RelationalEncoder.from_states(states)
    batch = encoder.transform(states)
    prior = TreePrior(max_depth=2)
    kernel = RevisionKernel(batch, RuleCatalog(encoder), prior)
    trees = enumerate_trees(batch, kernel.rules, prior)
    assert any(
        not p.impossible
        and p.move == "swap"
        and p.candidate.structure_key() != t.structure_key()
        for t in trees
        for p in kernel.swap_events(t)
    )
    assert any(
        p.impossible for t in trees if not t.is_leaf for p in kernel.change_events(t)
    )


def test_revision_detailed_balance_including_nontrivial_swaps():
    from test_bayesian_kernel import Fact
    from bayesian_rrtl.rules import RuleCatalog

    states = [
        [Fact("logical", str(a), "a", "o"), Fact("logical", str(b), "b", "o")]
        for a, b in [(False, False), (False, True), (True, False), (True, True)]
    ]
    encoder = RelationalEncoder.from_states(states)
    batch = encoder.transform(states)
    rules = RuleCatalog(encoder)
    prior = TreePrior(max_depth=2, alpha_tree=0.6)
    trees = enumerate_trees(batch, rules, prior)
    index = {t.structure_key(): i for i, t in enumerate(trees)}
    kernel = RevisionKernel(batch, rules, prior)
    y = np.array([-0.2, 0.1, 0.2, -0.1])
    pi = exact_posterior(trees, batch, y, 0.1, 0.0625, prior, rules)
    # Balance per move, stronger than checking the combined kernel only.
    for move in ("grow", "prune", "change", "swap"):
        reverse_move = {"grow": "prune", "prune": "grow"}.get(move, move)
        forward = np.zeros((len(trees), len(trees)))
        reverse = np.zeros_like(forward)
        for i, t in enumerate(trees):
            for p in kernel.all_proposals(t):
                if p.impossible or p.move not in (move, reverse_move):
                    continue
                j = index[p.candidate.structure_key()]
                probability = math.exp(
                    p.log_q_forward
                    + min(
                        0,
                        log_acceptance_ratio(t, p, batch, y, 0.1, 0.0625, prior, rules),
                    )
                )
                if p.move == move:
                    forward[i, j] += pi[i] * probability
                if p.move == reverse_move:
                    reverse[i, j] += pi[i] * probability
        np.testing.assert_allclose(forward, reverse.T, atol=1e-14)
        if move == "swap":
            assert (forward - np.diag(np.diag(forward))).sum() > 0


def test_linear_conditional_matches_manual_rng_and_one_noise_update():
    from bayesian_rrtl.general_bart import GaussianNoise
    from bayesian_rrtl.priors import NoisePrior

    class CountingNoise:
        prior = NoisePrior()

        def __init__(self):
            self.calls = []

        def sample(self, residuals, rng):
            self.calls.append(residuals.copy())
            return GaussianNoise(self.prior).sample(residuals, rng)

    noise = CountingNoise()
    batch = RelationalEncoder([]).transform([[]] * 6)
    design = np.linspace(-1, 1, 6)[:, None]
    result = GeneralBART(BARTRegressor(m=2), LinearMean([[0.5]]), noise).fit(
        batch,
        design[:, 0] * 0.1,
        np.random.default_rng(5),
        design=design,
        burn_in=3,
        draws=7,
    )
    assert len(noise.calls) == 10 and len(result.sigma2) == 7
    linear = LinearMean([[0.5]])
    residuals = design[:, 0] * 0.1
    mean, chol = linear.conditional(design, residuals, 0.2)
    expected = mean + np.linalg.solve(chol.T, np.random.default_rng(25).normal(size=1))
    np.testing.assert_array_equal(
        linear.sample(design, residuals, 0.2, np.random.default_rng(25)), expected
    )


def test_direct_revision_sampler_matches_enumerated_event_probabilities():
    from collections import Counter

    batch, _, rules = tiny_data()
    prior = TreePrior(max_depth=2)
    kernel = RevisionKernel(batch, rules, prior)
    trees = enumerate_trees(batch, rules, prior)
    tree = next(t for t in trees if not t.is_leaf and not t.true_child.is_leaf)

    def event_key(p):
        return p.move, p.path, p.candidate.structure_key(), p.impossible

    expected = Counter()
    for p in kernel.all_proposals(tree):
        expected[event_key(p)] += math.exp(p.log_q_forward)
    rng = np.random.default_rng(501)
    n = 6000
    observed = Counter(event_key(kernel.propose(tree, rng)) for _ in range(n))
    assert set(observed) <= set(expected)
    assert sum(abs(observed[k] / n - p) for k, p in expected.items()) / 2 < 0.055


def test_fast_rule_counts_preserve_group_order_and_probabilities():
    from bayesian_rrtl.rules import RuleCatalog
    from test_bayesian_kernel import Fact

    encoder = RelationalEncoder.from_states(
        [[Fact("comparative", "less", "x", "o"), Fact("logical", "True", "p", "o")]]
    )
    states = [
        [],
        [Fact("comparative", "less", "x", "o")],
        [Fact("comparative", "more", "x", "o"), Fact("logical", "True", "p", "o")],
    ]
    batch = encoder.transform(states)
    rules = RuleCatalog(encoder)
    for rows in (np.arange(3), np.array([2, 0]), np.array([1])):
        subset = batch.take(rows)
        expected = {}
        for rule in rules.rules:
            count = np.count_nonzero(rule.route(subset))
            if min(count, len(rows) - count) >= 1:
                expected.setdefault(rule.relation, []).append(rule)
        expected = {key: tuple(value) for key, value in expected.items()}
        actual = rules.eligible(batch, rows)
        assert list(actual.items()) == list(expected.items())
