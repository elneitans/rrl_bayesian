import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import invgamma, multivariate_normal, norm

from bayesian_rrtl.likelihood import LeafStats
from bayesian_rrtl.priors import (FixedScale, NoisePrior, calibrate_noise,
                                  leaf_variance, sample_inverse_gamma)


@pytest.mark.parametrize('residuals,sigma2,tau2', [([], 0.3, 0.1), ([0.2], 0.2, 0.0625),
                                                 ([-0.4, 0.1, 0.5], 0.1, 0.0625),
                                                 ([3.0, 2.0], 2.0, 1.0)])
def test_marginal_against_integration_and_multivariate_normal(residuals, sigma2, tau2):
    r = np.array(residuals)
    stats = LeafStats.from_residuals(r)
    def integrand(mu):
        return np.exp(norm.logpdf(mu, scale=np.sqrt(tau2)) + norm.logpdf(r, mu, np.sqrt(sigma2)).sum())
    integral, error = quad(integrand, -np.inf, np.inf, epsabs=1e-12)
    assert stats.log_marginal(sigma2, tau2) == pytest.approx(np.log(integral), abs=1e-9)
    if len(r):
        covariance = sigma2 * np.eye(len(r)) + tau2 * np.ones((len(r), len(r)))
        assert stats.log_marginal(sigma2, tau2) == pytest.approx(multivariate_normal.logpdf(r, cov=covariance))
    else:
        assert stats.posterior(sigma2, tau2) == (0, tau2)
        assert stats.log_marginal(sigma2, tau2) == 0


def test_leaf_draws_match_conjugate_moments():
    stats = LeafStats.from_residuals([0.1, 0.3, -0.2, 0.5])
    sigma2, tau2 = 0.2, 0.0625
    mean, variance = stats.posterior(sigma2, tau2)
    assert mean == pytest.approx(tau2 * stats.total / (sigma2 + stats.n * tau2))
    n = 100000
    draws = np.random.default_rng(900).normal(mean, np.sqrt(variance), n)
    assert abs(draws.mean() - mean) < 6 * np.sqrt(variance / n)
    assert abs(draws.var(ddof=1) - variance) < 6 * variance * np.sqrt(2 / (n - 1))


def test_noise_conditional_and_draw_moments():
    residuals = np.linspace(-0.4, 0.4, 30)
    prior = NoisePrior(nu=3, lambda_=0.08)
    shape, scale = prior.conditional(residuals)
    assert shape == 16.5  # No additional leaf-prior terms.
    assert scale == pytest.approx((0.24 + residuals @ residuals) / 2)
    n = 150000
    draws = sample_inverse_gamma(shape, scale, np.random.default_rng(903), size=n)
    dist = invgamma(a=shape, scale=scale)
    assert abs(draws.mean() - dist.mean()) < 6 * np.sqrt(dist.var() / n)
    fourth_central = (dist.moment(4) - 4 * dist.mean() * dist.moment(3)
                      + 6 * dist.mean()**2 * dist.moment(2) - 3 * dist.mean()**4)
    assert abs(draws.var() - dist.var()) < 6 * np.sqrt((fourth_central - dist.var()**2) / n)


def test_fixed_scale_and_noise_pilot_calibration():
    scale = FixedScale.from_rewards(-0.9, 1.1, 0.99)
    assert scale.lower == pytest.approx(-90)
    assert scale.upper == pytest.approx(110)
    values = np.array([-150, 0, 250])
    np.testing.assert_allclose(scale.restore(scale.normalize(values)), values)
    assert scale.normalize(values)[-1] > 0.5  # No clipping.
    y = np.array([0.1, 0.4, -0.2, 0.2, 0.5])
    design = np.column_stack([np.ones(5), np.arange(5)])
    calibration = calibrate_noise(y, design)
    assert calibration.method == 'linear_residuals'
    assert invgamma.cdf(calibration.residual_variance, calibration.prior.nu / 2,
                        scale=calibration.prior.nu * calibration.prior.lambda_ / 2) == pytest.approx(0.9)
    assert calibrate_noise([0]).method == 'insufficient_data_floor'
    assert 'rank_fallback' in calibrate_noise(y, np.ones((5, 2))).method
    assert leaf_variance(2, 1) == 0.0625
    assert leaf_variance(2, 20) == pytest.approx(0.0625 / 20)
