"""Structure, leaf and noise priors, with explicit fixed response scaling."""

from dataclasses import dataclass
import math

import numpy as np
from scipy.stats import chi2

from .likelihood import positive, residual_vector


@dataclass(frozen=True)
class FixedScale:
    lower: float = -0.5
    upper: float = 0.5

    def __post_init__(self):
        if not math.isfinite(self.lower) or not math.isfinite(self.upper) or self.upper <= self.lower:
            raise ValueError("Scale needs finite lower < upper")

    @property
    def c(self):
        return (self.lower + self.upper) / 2

    @property
    def width(self):
        return self.upper - self.lower

    def normalize(self, y):
        return (np.asarray(y, dtype=float) - self.c) / self.width

    def restore(self, y):
        return self.c + self.width * np.asarray(y, dtype=float)

    @classmethod
    def from_rewards(cls, r_min, r_max, gamma):
        if not (math.isfinite(r_min) and math.isfinite(r_max) and r_min <= r_max and 0 <= gamma < 1):
            raise ValueError("Invalid reward bounds or discount")
        return cls(min(0.0, r_min / (1 - gamma)), max(0.0, r_max / (1 - gamma)))


@dataclass(frozen=True)
class TreePrior:
    alpha_tree: float = 0.95
    beta_tree: float = 2.0
    max_depth: int = 6
    min_child: int = 1

    def __post_init__(self):
        if not 0 < self.alpha_tree < 1 or not math.isfinite(self.beta_tree) or self.beta_tree < 0:
            raise ValueError("Invalid depth prior")
        if type(self.max_depth) is not int or self.max_depth < 0 or type(self.min_child) is not int or self.min_child < 1:
            raise ValueError("Invalid tree support")

    def eligible(self, rules, batch, indices, depth):
        return rules.eligible(batch, indices, depth=depth, max_depth=self.max_depth, min_child=self.min_child)

    def split_probability(self, depth, eligible):
        return self.alpha_tree * (1 + depth)**(-self.beta_tree) if eligible else 0.0

    def log_prior(self, tree, batch, rules):
        def visit(node, rows, depth):
            eligible = self.eligible(rules, batch, rows, depth)
            p_split = self.split_probability(depth, eligible)
            if node.is_leaf:
                return math.log1p(-p_split)
            if node.rule not in eligible.get(node.rule.relation, ()):
                return -math.inf
            route = node.rule.route(batch)[rows]
            return (math.log(p_split) + rules.log_probability(node.rule, eligible)
                    + visit(node.true_child, rows[route], depth + 1)
                    + visit(node.false_child, rows[~route], depth + 1))
        return visit(tree, np.arange(len(batch)), 0)


def leaf_variance(k=2.0, m=1):
    positive(k, "k")
    if type(m) is not int or m < 1:
        raise ValueError("m must be a positive integer")
    return (0.5 / (k * math.sqrt(m)))**2


def sample_inverse_gamma(shape, scale, rng, size=None):
    """Density proportional to z**(-shape-1) * exp(-scale/z)."""
    positive(shape, "shape")
    positive(scale, "scale")
    draw = 1.0 / rng.gamma(shape=shape, scale=1.0 / scale, size=size)
    if not np.isfinite(draw).all() or np.any(np.asarray(draw) <= 0):
        raise FloatingPointError("Non-finite inverse-gamma draw")
    return draw


@dataclass(frozen=True)
class NoisePrior:
    nu: float = 3.0
    lambda_: float = 0.01

    def __post_init__(self):
        positive(self.nu, "nu")
        positive(self.lambda_, "lambda")

    def conditional(self, residuals):
        residuals = residual_vector(residuals)
        return (self.nu + len(residuals)) / 2, (self.nu * self.lambda_ + residuals @ residuals) / 2

    def sample(self, residuals, rng):
        return float(sample_inverse_gamma(*self.conditional(residuals), rng))


@dataclass(frozen=True)
class NoiseCalibration:
    prior: NoisePrior
    residual_variance: float
    method: str
    variance_floor: float
    quantile: float


def calibrate_noise(y_normalized, design=None, *, nu=3.0, quantile=0.9, variance_floor=1e-8):
    """Call once on training pilot data, already on the fixed response scale.

    A supplied full-rank design is used as-is (include an intercept if needed).
    Otherwise use sample response variance and an explicit positive floor.
    """
    y = residual_vector(y_normalized)
    positive(nu, "nu")
    positive(variance_floor, "variance_floor")
    if not 0 < quantile < 1:
        raise ValueError("quantile must lie in (0, 1)")
    variance = float(np.var(y, ddof=1)) if len(y) > 1 else 0.0
    method = "response_variance" if len(y) > 1 else "insufficient_data"
    if design is not None:
        design = np.asarray(design, dtype=float)
        if design.ndim != 2 or len(design) != len(y) or not np.isfinite(design).all():
            raise ValueError("Invalid pilot design")
        n, p = design.shape
        if n > p > 0 and np.linalg.matrix_rank(design) == p:
            residual = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
            variance = float(residual @ residual / (n - p))
            method = "linear_residuals"
        else:
            method += "_rank_fallback"
    if variance < variance_floor:
        variance = variance_floor
        method += "_floor"
    prior = NoisePrior(nu, variance * chi2.ppf(1 - quantile, nu) / nu)
    return NoiseCalibration(prior, variance, method, variance_floor, quantile)
