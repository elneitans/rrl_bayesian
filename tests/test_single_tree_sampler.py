import numpy as np
import pytest

from bayesian_rrtl import RelationalEncoder
from bayesian_rrtl.exact import enumerate_trees, exact_posterior, transition_matrix
from bayesian_rrtl.priors import FixedScale, NoisePrior, TreePrior
from bayesian_rrtl.sampler import SingleTreeSampler
from test_bayesian_kernel import tiny_data


def test_mcmc_frequencies_match_exact_posterior_with_mc_tolerance():
    batch, y, rules = tiny_data()
    prior = TreePrior(alpha_tree=0.6, beta_tree=1, max_depth=2)
    trees = enumerate_trees(batch, rules, prior)
    probability = exact_posterior(trees, batch, y, 0.1, 0.0625, prior, rules)
    matrix = transition_matrix(trees, batch, y, 0.1, 0.0625, prior, rules)
    # Reversible spectral bound on indicator autocorrelation; not an iid error bar.
    symmetric = np.sqrt(probability[:, None]) * matrix / np.sqrt(probability[None, :])
    rho = np.max(np.abs(np.linalg.eigvalsh(symmetric)[:-1]))
    n = 10000
    posterior = SingleTreeSampler(tree_prior=prior).fit(batch, y, np.random.default_rng(1709),
                                                        burn_in=1000, draws=n, fixed_sigma2=0.1)
    index = {tree.structure_key(): i for i, tree in enumerate(trees)}
    counts = np.bincount([index[t.structure_key()] for t in posterior.trees], minlength=len(trees))
    variance_bound = probability * (1 - probability) * (1 + rho) / (1 - rho) / n
    assert np.all(np.abs(counts / n - probability) < 6 * np.sqrt(variance_bound) + 1/n)
    assert (np.abs(counts / n - probability).sum() / 2) < 0.07


def test_unknown_noise_root_chain_matches_conditional_integration():
    # With mu tightly identified, verify the full Gibbs chain against independent
    # numerical quadrature of the joint root posterior, not just an RNG primitive.
    from scipy.integrate import quad
    from scipy.stats import norm
    encoder = RelationalEncoder([])
    batch = encoder.transform([[]] * 8)
    y = np.array([-0.12, -0.02, 0.0, 0.04, 0.1, 0.18, 0.2, 0.25])
    prior = NoisePrior(nu=8, lambda_=0.04)
    tau2 = 0.0625
    power = (prior.nu + len(y)) / 2
    def weight(mu):
        return norm.pdf(mu, scale=np.sqrt(tau2)) * (prior.nu * prior.lambda_ + np.sum((y-mu)**2))**(-power)
    normalizer = quad(weight, -np.inf, np.inf)[0]
    mean_mu = quad(lambda mu: mu * weight(mu), -np.inf, np.inf)[0] / normalizer
    mean_sigma2 = quad(lambda mu: (prior.nu * prior.lambda_ + np.sum((y-mu)**2)) / (prior.nu + len(y)-2)
                      * weight(mu), -np.inf, np.inf)[0] / normalizer
    posterior = SingleTreeSampler(tree_prior=TreePrior(max_depth=0), noise_prior=prior).fit(
        batch, y, np.random.default_rng(222), burn_in=500, draws=8000)
    draws = posterior.predict_latent_draws(batch)[:, 0]
    # Batch means estimate MC standard errors while allowing within-chain correlation.
    for samples, target in [(draws, mean_mu), (np.array(posterior.sigma2), mean_sigma2)]:
        batches = samples.reshape(40, -1).mean(axis=1)
        se = batches.std(ddof=1) / np.sqrt(len(batches))
        assert abs(samples.mean() - target) < 6 * se
    assert np.isfinite(posterior.sigma2).all()


def test_prediction_units_reproducibility_and_input_validation():
    batch, y, _ = tiny_data()
    sampler = SingleTreeSampler(tree_prior=TreePrior(max_depth=1), scale=FixedScale(10, 30))
    a = sampler.fit(batch, y * 20 + 20, np.random.default_rng(8), burn_in=10, draws=50)
    b = sampler.fit(batch, y * 20 + 20, np.random.default_rng(8), burn_in=10, draws=50)
    np.testing.assert_array_equal(a.predict_latent_draws(batch), b.predict_latent_draws(batch))
    assert a.trace == b.trace
    assert a.sigma2 == b.sigma2
    latent = a.predict_latent_draws(batch)
    np.testing.assert_allclose(a.predict_mean(batch), latent.mean(axis=0))
    assert latent.shape == (50, 3)
    rng1, rng2 = np.random.default_rng(42), np.random.default_rng(42)
    np.testing.assert_allclose(a.predict_observation_draws(batch, rng1),
                               latent + rng2.normal(size=latent.shape) * np.sqrt(a.sigma2)[:, None] * 20)
    assert a.predict_mean(batch.take([])).shape == (0,)
    with pytest.raises(ValueError, match='different catalog'):
        a.predict_mean(RelationalEncoder([]).transform([[]]))
    for bad in [[], [1, 2], [1, np.nan, 2]]:
        with pytest.raises(ValueError):
            sampler.fit(batch, bad, np.random.default_rng(1))
    out = sampler.fit(batch, [0, 20, 100], np.random.default_rng(2), burn_in=0, draws=1)
    assert out.targets_outside_scale == 2
