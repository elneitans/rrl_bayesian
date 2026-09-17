"""Single-tree Gaussian Bayesian regression, on fixed data (H2, m=1)."""

from dataclasses import dataclass
import hashlib
import math

import numpy as np

from .encoding import EncodedBatch, RelationalEncoder
from .likelihood import LeafStats, positive, residual_vector
from .priors import FixedScale, NoisePrior, TreePrior, leaf_variance
from .proposals import GrowPruneKernel, log_acceptance_ratio
from .rules import RuleCatalog
from .tree import Tree


@dataclass(frozen=True)
class SweepDiagnostic:
    move: str
    accepted: bool
    impossible: bool
    log_ratio: float
    leaves: int
    depth: int
    sigma2: float
    residual_sum_squares: float


@dataclass(frozen=True)
class SingleTreePosterior:
    trees: tuple[Tree, ...]
    sigma2: tuple[float, ...]
    catalog_hash: str
    scale: FixedScale
    tree_prior: TreePrior
    noise_prior: NoisePrior
    k: float
    burn_in: int
    fixed_sigma2: float | None
    data_hash: str
    trace: tuple[SweepDiagnostic, ...]
    targets_outside_scale: int

    def predict_latent_draws(self, batch):
        if batch.catalog_hash != self.catalog_hash:
            raise ValueError("Prediction batch uses a different catalog")
        return self.scale.restore(np.stack([tree.predict(batch) for tree in self.trees]))

    def predict_mean(self, batch):
        return self.predict_latent_draws(batch).mean(axis=0)

    def predict_observation_draws(self, batch, rng):
        latent = self.predict_latent_draws(batch)
        noise = rng.normal(size=latent.shape) * np.sqrt(self.sigma2)[:, None] * self.scale.width
        return latent + noise

    def diagnostics(self):
        result = {"sweeps": len(self.trace), "burn_in": self.burn_in, "retained": len(self.trees),
                  "targets_outside_scale": self.targets_outside_scale, "moves": {}}
        for move in ("grow", "prune"):
            steps = [step for step in self.trace if step.move == move]
            possible = sum(not step.impossible for step in steps)
            accepted = sum(step.accepted for step in steps)
            result["moves"][move] = {"selected": len(steps), "impossible": len(steps) - possible,
                                     "accepted": accepted, "rejected": possible - accepted,
                                     "acceptance_given_possible": accepted / possible if possible else None}
        return result


def sample_leaves(tree, batch, residuals, sigma2, tau2, rng):
    """Fresh Gibbs update on every leaf, even after a rejected/impossible move."""
    result = tree
    for path, (leaf, rows) in tree.partition(batch).items():
        mean, variance = LeafStats.from_residuals(residuals[rows]).posterior(sigma2, tau2)
        result = result.replace(path, Tree(float(rng.normal(mean, math.sqrt(variance)))))
    return result


@dataclass(frozen=True)
class TreeUpdate:
    move: str
    accepted: bool
    impossible: bool
    log_ratio: float


def update_tree(tree, sigma2, batch, residuals, kernel, rng, tau2, tree_prior):
    """Collapsed MH and leaf Gibbs update; never updates shared residual noise."""
    proposal = kernel.propose(tree, rng)
    log_ratio = log_acceptance_ratio(tree, proposal, batch, residuals, sigma2,
                                     tau2, tree_prior, kernel.rules)
    accepted = False
    if not proposal.impossible:
        uniform = float(rng.random())
        log_uniform = math.log(uniform) if uniform else -math.inf
        accepted = log_uniform < min(0.0, log_ratio)
    selected = proposal.candidate if accepted else tree
    selected = sample_leaves(selected, batch, residuals, sigma2, tau2, rng)
    return selected, TreeUpdate(proposal.move, accepted, proposal.impossible, log_ratio)


class SingleTreeSampler:
    def __init__(self, *, tree_prior=TreePrior(), noise_prior=NoisePrior(), scale=FixedScale(), k=2.0):
        positive(k, "k")
        self.tree_prior, self.noise_prior, self.scale, self.k = tree_prior, noise_prior, scale, k
        self.tau2 = leaf_variance(k, m=1)

    def sweep(self, tree, sigma2, batch, y_normalized, kernel, rng, *, fixed_sigma2=None):
        selected, update = update_tree(tree, sigma2, batch, y_normalized, kernel, rng,
                                       self.tau2, self.tree_prior)
        residuals = y_normalized - selected.predict(batch)
        sigma2 = (self.noise_prior.sample(residuals, rng) if fixed_sigma2 is None else fixed_sigma2)
        partitions = selected.partition(batch)
        diagnostic = SweepDiagnostic(update.move, update.accepted, update.impossible, update.log_ratio,
                                     len(partitions), max(map(len, partitions)), float(sigma2),
                                     float(residuals @ residuals))
        return selected, float(sigma2), diagnostic

    def fit(self, batch, y, rng, *, burn_in=500, draws=1000, fixed_sigma2=None):
        """Restart a chain; no reusing posterior draws or changing X/y inside it.

        fixed_sigma2 is a normalized variance, for mathematical validation only.
        Predictions and supplied responses use original units; leaves/noise use
        normalized units. This is a single tree, not the additive H3 ensemble.
        """
        if type(burn_in) is not int or burn_in < 0 or type(draws) is not int or draws < 1:
            raise ValueError("Require nonnegative burn_in and positive draws")
        y = residual_vector(y).copy()
        if not len(y) or len(y) != len(batch):
            raise ValueError("fit requires nonempty aligned X and y")
        if fixed_sigma2 is not None:
            positive(fixed_sigma2, "fixed_sigma2")
        # Own the arrays so callers cannot alter data while the chain is running.
        batch = EncodedBatch(batch.values, batch.observed, batch.catalog, batch.catalog_hash)
        rules = RuleCatalog(RelationalEncoder(batch.catalog))
        normalized = self.scale.normalize(y)
        normalized.flags.writeable = False
        data_hash = hashlib.sha256(batch.catalog_hash.encode() + batch.values.tobytes()
                                   + batch.observed.tobytes() + y.tobytes()).hexdigest()
        kernel = GrowPruneKernel(batch, rules, self.tree_prior)
        tree = Tree(float(normalized.mean()))
        sigma2 = fixed_sigma2 if fixed_sigma2 is not None else self.noise_prior.lambda_
        trees, variances, trace = [], [], []
        for sweep in range(burn_in + draws):
            tree, sigma2, diagnostic = self.sweep(tree, sigma2, batch, normalized, kernel, rng,
                                                 fixed_sigma2=fixed_sigma2)
            trace.append(diagnostic)
            if sweep >= burn_in:
                trees.append(tree)
                variances.append(sigma2)
        return SingleTreePosterior(tuple(trees), tuple(variances), batch.catalog_hash,
                                   self.scale, self.tree_prior, self.noise_prior, self.k,
                                   burn_in, fixed_sigma2, data_hash, tuple(trace),
                                   int(np.count_nonzero((y < self.scale.lower) | (y > self.scale.upper))))
