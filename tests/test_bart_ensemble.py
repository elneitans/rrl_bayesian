from dataclasses import replace

import numpy as np
import pytest

from bayesian_rrtl import RelationalEncoder, Tree
from bayesian_rrtl.ensemble import BARTRegressor, EnsembleDraw, PredictionCache
from bayesian_rrtl.exact import enumerate_trees, exact_ensemble_posterior
from bayesian_rrtl.priors import FixedScale, TreePrior
from bayesian_rrtl.sampler import SingleTreeSampler, TreeUpdate
from test_bayesian_kernel import tiny_data


@pytest.mark.parametrize('sigma2', [None, 0.1])
def test_m_one_reduces_to_h2_draw_for_draw(sigma2):
    batch, y, _ = tiny_data()
    options = dict(tree_prior=TreePrior(max_depth=2), scale=FixedScale(-0.5, 0.5))
    single = SingleTreeSampler(**options).fit(batch, y, np.random.default_rng(71),
                                               burn_in=30, draws=100, fixed_sigma2=sigma2)
    ensemble = BARTRegressor(m=1, **options).fit(batch, y, np.random.default_rng(71),
                                                burn_in=30, draws=100, fixed_sigma2=sigma2, verify_cache=True)
    np.testing.assert_array_equal(ensemble.predict_latent_draws(batch), single.predict_latent_draws(batch))
    assert ensemble.sigma2 == single.sigma2
    assert [d.trees[0] for d in ensemble.draws] == list(single.trees)
    assert ensemble.max_cache_error == 0


def test_sequential_residuals_and_one_noise_update_per_sweep(monkeypatch):
    batch = RelationalEncoder([]).transform([[], []])
    original = [Tree(0.1), Tree(0.2), Tree(0.3)]
    cache = PredictionCache(original, batch)
    calls, noise_calls = [], []
    def fake_update(tree, sigma2, x, residuals, kernel, rng, tau2, tree_prior):
        calls.append(residuals.copy())
        assert sigma2 == 0.1
        assert tau2 == pytest.approx(0.0625 / 3)
        return Tree([0.4, 0.5, 0.6][len(calls)-1]), TreeUpdate('grow', False, True, 0)
    class Noise:
        def sample(self, residuals, rng):
            noise_calls.append(residuals.copy())
            return 0.7
    monkeypatch.setattr('bayesian_rrtl.ensemble.update_tree', fake_update)
    sigma2, diagnostic, error = BARTRegressor(m=3, noise_prior=Noise()).sweep(
        cache, 0.1, np.array([1., 2.]), None, np.random.default_rng(1), verify_cache=True)
    np.testing.assert_allclose(calls, [[0.5, 1.5], [0.3, 1.3], [0.1, 1.1]])
    np.testing.assert_allclose(noise_calls, [[-0.5, 0.5]])
    assert sigma2 == 0.7 and diagnostic.residual_sum_squares == pytest.approx(0.5)
    assert diagnostic.total_leaves == 3 and diagnostic.max_depth == 0
    assert error < 1e-15
    assert original == [Tree(0.1), Tree(0.2), Tree(0.3)]


def test_sum_inside_draw_mean_across_draws_and_restore_scale_once():
    batch, y, _ = tiny_data()
    posterior = BARTRegressor(m=2, scale=FixedScale(10, 30)).fit(
        batch, y, np.random.default_rng(2), burn_in=0, draws=2)
    posterior = replace(posterior, draws=(EnsembleDraw((Tree(0.1), Tree(0.2)), 0.01),
                                          EnsembleDraw((Tree(0.4), Tree(0.6)), 0.04)))
    latent = posterior.predict_latent_draws(batch)
    np.testing.assert_allclose(latent, [[26]*3, [40]*3])
    np.testing.assert_allclose(posterior.predict_mean(batch), [33]*3)
    noise = np.random.default_rng(123).normal(size=(2, 3)) * np.sqrt([0.01, 0.04])[:, None] * 20
    np.testing.assert_allclose(posterior.predict_observation_draws(batch, np.random.default_rng(123)), latent+noise)
    assert posterior.predict_mean(batch.take([])).shape == (0,)
    with pytest.raises(ValueError, match='different catalog'):
        posterior.predict_mean(RelationalEncoder([]).transform([[]]))


def test_snapshots_data_hash_reproducibility_and_cache_audit():
    batch, y, _ = tiny_data()
    original_values = batch.values.copy()
    original_y = y.copy()
    model = BARTRegressor(m=3, tree_prior=TreePrior(max_depth=2))
    a = model.fit(batch, y, np.random.default_rng(15), burn_in=10, draws=80, verify_cache=True)
    saved = a.predict_latent_draws(batch).copy()
    b = model.fit(batch, y, np.random.default_rng(15), burn_in=10, draws=80)
    assert a.draws == b.draws and a.trace == b.trace
    model.fit(batch, y + 0.3, np.random.default_rng(3), burn_in=0, draws=10)
    np.testing.assert_array_equal(a.predict_latent_draws(batch), saved)
    np.testing.assert_array_equal(batch.values, original_values)
    np.testing.assert_array_equal(y, original_y)
    assert a.max_cache_error < 1e-12
    assert b.max_cache_error is None
    assert sum(v['selected'] for v in a.diagnostics()['moves'].values()) == 3 * 90
    assert a.data_hash == b.data_hash


def test_full_ensemble_matches_independently_integrated_gaussian_reference():
    batch, y, rules = tiny_data()
    prior = TreePrior(alpha_tree=0.6, max_depth=1)
    model = BARTRegressor(m=2, tree_prior=prior)
    trees = enumerate_trees(batch, rules, prior)
    ensembles, probabilities, mean, covariance = exact_ensemble_posterior(
        trees, 2, batch, y, 0.1, model.tau2, prior, rules)
    posterior = model.fit(batch, y, np.random.default_rng(803), burn_in=1000, draws=12000, fixed_sigma2=0.1)
    latent = posterior.predict_latent_draws(batch)
    blocks = latent.reshape(40, -1, len(batch)).mean(axis=1)
    se = blocks.std(axis=0, ddof=1) / np.sqrt(40)
    assert np.all(np.abs(latent.mean(axis=0) - mean) < 6 * se)
    variance_blocks = ((latent - mean)**2).reshape(40, -1, len(batch)).mean(axis=1)
    se_variance = variance_blocks.std(axis=0, ddof=1) / np.sqrt(40)
    assert np.all(np.abs(variance_blocks.mean(axis=0) - np.diag(covariance)) < 6 * se_variance)
    index = {tuple(t.structure_key() for t in ensemble): i for i, ensemble in enumerate(ensembles)}
    ids = [index[tuple(t.structure_key() for t in draw.trees)] for draw in posterior.draws]
    frequencies = np.bincount(ids, minlength=len(ensembles)) / len(ids)
    assert np.abs(frequencies - probabilities).sum() / 2 < 0.06


@pytest.mark.parametrize('m', [1, 4])
def test_root_ensemble_has_correct_total_prior_shrinkage(m):
    # The sum of m independent leaf priors has variance .0625, independent of m.
    batch = RelationalEncoder([]).transform([[]] * 3)
    y, sigma2 = np.array([-0.1, 0.1, 0.3]), 0.2
    variance = 1 / (3 / sigma2 + 1 / 0.0625)
    mean = variance * y.sum() / sigma2
    posterior = BARTRegressor(m=m, tree_prior=TreePrior(max_depth=0)).fit(
        batch, y, np.random.default_rng(331), burn_in=200, draws=5000, fixed_sigma2=sigma2)
    samples = posterior.predict_latent_draws(batch)[:, 0]
    blocks = samples.reshape(25, -1).mean(axis=1)
    assert abs(samples.mean() - mean) < 6 * blocks.std(ddof=1) / 5
    assert abs(samples.var() - variance) < 0.004


def test_invalid_fit_and_ensemble_settings():
    batch, y, _ = tiny_data()
    for m in [0, -1, 1.5, True]:
        with pytest.raises(ValueError):
            BARTRegressor(m=m)
    for bad in [[], [1, 2], [1, np.nan, 2]]:
        with pytest.raises(ValueError):
            BARTRegressor(m=2).fit(batch, bad, np.random.default_rng(1))
    with pytest.raises(ValueError):
        BARTRegressor().fit(batch, y, np.random.default_rng(1), fixed_sigma2=0)
    with pytest.raises(ValueError):
        EnsembleDraw((), 1)
    with pytest.raises(ValueError):
        EnsembleDraw((Tree(),), -1)
    posterior = BARTRegressor(m=2).fit(batch, [0, 1, -1], np.random.default_rng(2), burn_in=0, draws=1)
    assert posterior.targets_outside_scale == 2


def test_default_twenty_tree_ensemble_cache_and_retained_structure():
    batch, y, _ = tiny_data()
    posterior = BARTRegressor().fit(batch, y, np.random.default_rng(21),
                                    burn_in=5, draws=20, verify_cache=True)
    assert posterior.m == 20
    assert all(len(draw.trees) == 20 for draw in posterior.draws)
    assert all(len(step.updates) == 20 for step in posterior.trace)
    assert posterior.max_cache_error < 1e-12
    assert np.isfinite(posterior.predict_latent_draws(batch)).all()
