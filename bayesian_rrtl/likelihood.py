"""Gaussian leaf likelihood with leaf means integrated out (independent prior)."""

from dataclasses import dataclass
import math

import numpy as np


def positive(value, name):
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")


def residual_vector(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Residuals must be a finite one-dimensional vector")
    return values


@dataclass(frozen=True)
class LeafStats:
    n: int
    total: float
    sum_squares: float

    def __post_init__(self):
        if type(self.n) is not int or self.n < 0:
            raise ValueError("n must be a nonnegative integer")
        if not math.isfinite(self.total) or not math.isfinite(self.sum_squares) or self.sum_squares < 0:
            raise ValueError("Sufficient statistics must be finite with nonnegative sum of squares")
        if not self.n and (self.total != 0 or self.sum_squares != 0):
            raise ValueError("Empty leaf must have zero sufficient statistics")

    @classmethod
    def from_residuals(cls, residuals):
        residuals = residual_vector(residuals)
        return cls(len(residuals), float(residuals.sum()), float(residuals @ residuals))

    def posterior(self, sigma2, tau2):
        positive(sigma2, "sigma2")
        positive(tau2, "tau2")
        variance = 1.0 / (self.n / sigma2 + 1.0 / tau2)
        return variance * self.total / sigma2, variance

    def log_marginal(self, sigma2, tau2):
        positive(sigma2, "sigma2")
        positive(tau2, "tau2")
        return (-0.5 * self.n * math.log(2 * math.pi * sigma2)
                - 0.5 * math.log1p(self.n * tau2 / sigma2)
                - self.sum_squares / (2 * sigma2)
                + tau2 * self.total**2 / (2 * sigma2 * (sigma2 + self.n * tau2)))


def tree_log_marginal(tree, batch, residuals, sigma2, tau2):
    residuals = residual_vector(residuals)
    if len(residuals) != len(batch):
        raise ValueError("Residual and batch lengths differ")
    return sum(LeafStats.from_residuals(residuals[rows]).log_marginal(sigma2, tau2)
               for leaf, rows in tree.partition(batch).values())
