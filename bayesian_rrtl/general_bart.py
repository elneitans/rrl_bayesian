"""General BART with additive Gaussian linear mean, explicit normalized units."""

from dataclasses import dataclass

import numpy as np

from .encoding import RelationalEncoder
from .ensemble import BARTRegressor, PredictionCache
from .likelihood import positive, residual_vector
from .priors import NoisePrior
from .proposals import GrowPruneKernel, RevisionKernel
from .rules import RuleCatalog
from .tree import Tree


class ZeroMean:
    def fit(self, model, batch, y, rng, **options):
        # Exact reduction, including RNG consumption, to the H3 engine.
        return model.fit(batch, y, rng, **options)


@dataclass(frozen=True)
class GaussianNoise:
    prior: NoisePrior = NoisePrior()

    def sample(self, residuals, rng):
        return self.prior.sample(residuals, rng)


class LinearMean:
    """theta ~ N(0,V0); coefficients contribute in normalized response units."""

    def __init__(self, covariance):
        covariance = np.asarray(covariance, dtype=float)
        if (
            covariance.ndim != 2
            or covariance.shape[0] != covariance.shape[1]
            or not len(covariance)
        ):
            raise ValueError("Prior covariance must be a nonempty square matrix")
        if not np.isfinite(covariance).all() or not np.allclose(
            covariance, covariance.T
        ):
            raise ValueError("Prior covariance must be finite and symmetric")
        np.linalg.cholesky(covariance)
        self.covariance = covariance.copy()
        self.precision = np.linalg.solve(covariance, np.eye(len(covariance)))

    def conditional(self, design, residuals, sigma2):
        positive(sigma2, "sigma2")
        precision = self.precision + design.T @ design / sigma2
        chol = np.linalg.cholesky(precision)
        mean = np.linalg.solve(
            chol.T, np.linalg.solve(chol, design.T @ residuals / sigma2)
        )
        return mean, chol

    def sample(self, design, residuals, sigma2, rng):
        mean, chol = self.conditional(design, residuals, sigma2)
        return mean + np.linalg.solve(chol.T, rng.normal(size=len(mean)))


@dataclass(frozen=True)
class GeneralPosterior:
    trees: tuple
    theta: np.ndarray
    sigma2: np.ndarray
    center: np.ndarray
    scale: object
    catalog_hash: str

    def predict_latent_draws(self, batch, design):
        design = np.asarray(design, dtype=float)
        if batch.catalog_hash != self.catalog_hash or design.shape != (
            len(batch),
            len(self.center),
        ):
            raise ValueError("Prediction design/catalog mismatch")
        if not np.isfinite(design).all():
            raise ValueError("Design must be finite")
        f = np.array(
            [
                sum((t.predict(batch) for t in trees), np.zeros(len(batch)))
                for trees in self.trees
            ]
        )
        return self.scale.restore(f + self.theta @ (design - self.center).T)

    def predict_mean(self, batch, design):
        return self.predict_latent_draws(batch, design).mean(axis=0)

    def predict_observation_draws(self, batch, design, rng):
        latent = self.predict_latent_draws(batch, design)
        return (
            latent
            + rng.normal(size=latent.shape)
            * np.sqrt(self.sigma2)[:, None]
            * self.scale.width
        )


class GeneralBART:
    def __init__(self, bart=None, additional=None, noise=None):
        self.bart = bart or BARTRegressor()
        self.additional = additional or ZeroMean()
        self.noise = noise or GaussianNoise(self.bart.noise_prior)

    def fit(
        self,
        batch,
        y,
        rng,
        *,
        design=None,
        burn_in=500,
        draws=1000,
        fixed_sigma2=None,
        trees_enabled=True,
        center_design=True,
    ):
        if isinstance(self.additional, ZeroMean):
            if not trees_enabled:
                raise ValueError("ZeroMean requires enabled trees")
            if self.noise.prior != self.bart.noise_prior:
                raise ValueError("ZeroMean reduction requires matching noise priors")
            return self.additional.fit(
                self.bart,
                batch,
                y,
                rng,
                burn_in=burn_in,
                draws=draws,
                fixed_sigma2=fixed_sigma2,
            )
        if (
            type(burn_in) is not int
            or burn_in < 0
            or type(draws) is not int
            or draws < 1
        ):
            raise ValueError("Invalid chain lengths")
        if fixed_sigma2 is not None:
            positive(fixed_sigma2, "sigma2")
        y = residual_vector(y)
        design = np.array(design, dtype=float, copy=True)
        p = len(self.additional.covariance)
        if (
            not len(y)
            or len(y) != len(batch)
            or design.shape != (len(y), p)
            or not np.isfinite(design).all()
        ):
            raise ValueError("Invalid aligned response/design")
        center = design.mean(axis=0) if center_design else np.zeros(p)
        design -= center
        normalized = self.bart.scale.normalize(y)
        if not np.isfinite(normalized).all():
            raise ValueError("Non-finite normalized response")
        rules = RuleCatalog(RelationalEncoder(batch.catalog))
        kernel = (
            GrowPruneKernel if self.bart.kernel == "grow_prune" else RevisionKernel
        )(batch, rules, self.bart.tree_prior)
        cache = PredictionCache(
            [Tree(float(normalized.mean() / self.bart.m)) for _ in range(self.bart.m)],
            batch,
        )
        theta = np.zeros(p)
        sigma2 = self.noise.prior.lambda_ if fixed_sigma2 is None else fixed_sigma2
        trees, coefficients, variances = [], [], []
        for step in range(burn_in + draws):
            if trees_enabled:
                # Hold sigma fixed here; update it once after both additive means.
                self.bart.sweep(
                    cache,
                    sigma2,
                    normalized - design @ theta,
                    kernel,
                    rng,
                    fixed_sigma2=sigma2,
                )
                f = cache.total
            else:
                f = np.zeros(len(batch))
            theta = self.additional.sample(design, normalized - f, sigma2, rng)
            residuals = normalized - f - design @ theta
            sigma2 = (
                self.noise.sample(residuals, rng)
                if fixed_sigma2 is None
                else fixed_sigma2
            )
            if step >= burn_in:
                trees.append(tuple(cache.trees) if trees_enabled else ())
                coefficients.append(theta.copy())
                variances.append(sigma2)
        theta, sigma, center = (
            np.array(coefficients),
            np.array(variances),
            center.copy(),
        )
        theta.flags.writeable = sigma.flags.writeable = center.flags.writeable = False
        return GeneralPosterior(
            tuple(trees), theta, sigma, center, self.bart.scale, batch.catalog_hash
        )
