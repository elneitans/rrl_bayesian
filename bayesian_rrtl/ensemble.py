"""Gaussian BART on a fixed relational batch: sum trees, average posterior draws."""

from dataclasses import dataclass, replace
import hashlib
import time

import numpy as np

from .encoding import EncodedBatch, RelationalEncoder
from .likelihood import positive, residual_vector
from .priors import FixedScale, NoisePrior, TreePrior, leaf_variance
from .proposals import GrowPruneKernel, RevisionKernel
from .rules import RuleCatalog
from .sampler import TreeUpdate, update_tree
from .tree import Tree


class StructureLimitError(ValueError):
    """The new batch requires more support repairs than the allowed distance."""

    def __init__(self, tree_index, required_distance, limit):
        self.tree_index = tree_index
        self.required_distance = required_distance
        self.limit = limit
        super().__init__(f'Tree {tree_index} requires distance {required_distance}, limit {limit}')


def repair_tree_support(tree, batch, rules, prior):
    """Prune unsupported subtrees, preserving every supported ancestral split."""
    def visit(node, rows, depth):
        if node.is_leaf:
            return node
        eligible = prior.eligible(rules, batch, rows, depth)
        if node.rule not in eligible.get(node.rule.relation, ()):
            return Tree(float(node.predict(batch)[rows].mean()))
        route = node.rule.route(batch)[rows]
        return Tree(rule=node.rule, true_child=visit(node.true_child, rows[route], depth + 1),
                    false_child=visit(node.false_child, rows[~route], depth + 1))

    return visit(tree, np.arange(len(batch)), 0)


class BoundedStructureKernel:
    """Reject proposals outside a fixed neighbourhood of the inherited tree."""

    def __init__(self, kernel, anchor, limit):
        self.kernel, self.anchor, self.limit = kernel, anchor, limit
        self.rules = kernel.rules
        self.distance_rejections = 0
        self.max_distance = 0

    def propose(self, tree, rng):
        proposal = self.kernel.propose(tree, rng)
        if not proposal.impossible and proposal.candidate.structure_distance(self.anchor) > self.limit:
            self.distance_rejections += 1
            return replace(proposal, candidate=tree, impossible=True)
        return proposal


class PredictionCache:
    """Mutable working state, never retained in posterior snapshots."""

    def __init__(self, trees, batch):
        self.trees = list(trees)
        if not self.trees:
            raise ValueError('An ensemble needs at least one tree')
        self.batch = batch
        self.by_tree = np.stack([tree.predict(batch) for tree in trees])
        self.total = self.by_tree.sum(axis=0)

    def partial_residuals(self, y, j):
        return y - (self.total - self.by_tree[j])

    def replace(self, j, tree):
        other = self.total - self.by_tree[j]
        self.by_tree[j] = tree.predict(self.batch)
        self.total = other + self.by_tree[j]
        self.trees[j] = tree

    def verify(self):
        """Check after ANY tree update against full recomputation, in debug runs."""
        recomputed = np.stack([tree.predict(self.batch) for tree in self.trees])
        np.testing.assert_allclose(self.by_tree, recomputed, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(self.total, recomputed.sum(axis=0), rtol=1e-12, atol=1e-12)
        return float(np.max(np.abs(self.total - recomputed.sum(axis=0)), initial=0))


@dataclass(frozen=True)
class EnsembleDraw:
    trees: tuple[Tree, ...]
    sigma2: float

    def __post_init__(self):
        object.__setattr__(self, 'trees', tuple(self.trees))
        if not self.trees or not all(isinstance(tree, Tree) for tree in self.trees):
            raise ValueError('Each draw needs a nonempty tuple of trees')
        positive(self.sigma2, 'sigma2')

    def predict(self, batch):
        return np.sum([tree.predict(batch) for tree in self.trees], axis=0)


@dataclass(frozen=True)
class EnsembleSweep:
    updates: tuple[TreeUpdate, ...]
    sigma2: float
    residual_sum_squares: float
    total_leaves: int
    max_depth: int


@dataclass(frozen=True)
class BARTPosterior:
    draws: tuple[EnsembleDraw, ...]
    catalog_hash: str
    data_hash: str
    scale: FixedScale
    tree_prior: TreePrior
    noise_prior: NoisePrior
    k: float
    burn_in: int
    fixed_sigma2: float | None
    trace: tuple[EnsembleSweep, ...]
    targets_outside_scale: int
    elapsed_seconds: float
    max_cache_error: float | None
    reused_trees: int | None = None
    structure_limit: int | None = None
    structure_distances: tuple[int, ...] | None = None
    support_repairs: tuple[int, ...] | None = None
    distance_rejections: int | None = None

    @property
    def m(self):
        return len(self.draws[0].trees)

    @property
    def sigma2(self):
        """Residual variances in normalized units, aligned with draws."""
        return tuple(draw.sigma2 for draw in self.draws)

    def predict_latent_draws(self, batch):
        if batch.catalog_hash != self.catalog_hash:
            raise ValueError('Prediction batch uses a different catalog')
        # Restore the intercept/scale ONCE per ensemble, not once per tree.
        return self.scale.restore(np.stack([draw.predict(batch) for draw in self.draws]))

    def predict_mean(self, batch):
        return self.predict_latent_draws(batch).mean(axis=0)

    def predict_observation_draws(self, batch, rng):
        latent = self.predict_latent_draws(batch)
        return latent + rng.normal(size=latent.shape) * np.sqrt(self.sigma2)[:, None] * self.scale.width

    def diagnostics(self):
        result = {'m': self.m, 'sweeps': len(self.trace), 'burn_in': self.burn_in,
                  'retained': len(self.draws), 'targets_outside_scale': self.targets_outside_scale,
                  'elapsed_seconds': self.elapsed_seconds, 'max_cache_error': self.max_cache_error,
                  'moves': {}}
        if self.reused_trees is not None:
            result.update(reused_trees=self.reused_trees, reset_trees=self.m-self.reused_trees)
        if self.structure_limit is not None:
            result.update(structure_limit=self.structure_limit,
                          max_structure_distance_per_tree=self.structure_distances,
                          support_repairs_per_tree=self.support_repairs,
                          distance_rejections=self.distance_rejections)
        for move in ('grow', 'prune', 'change', 'swap'):
            updates = [u for step in self.trace for u in step.updates if u.move == move]
            possible = sum(not u.impossible for u in updates)
            accepted = sum(u.accepted for u in updates)
            result['moves'][move] = {'selected': len(updates), 'impossible': len(updates) - possible,
                                     'accepted': accepted, 'rejected': possible - accepted,
                                     'acceptance_given_possible': accepted / possible if possible else None}
        return result


class BARTRegressor:
    def __init__(self, *, m=20, tree_prior=TreePrior(), noise_prior=NoisePrior(),
                 scale=FixedScale(), k=2.0, kernel="grow_prune"):
        if kernel not in ("grow_prune", "revision"):
            raise ValueError("Unknown structural kernel")
        self.kernel = kernel
        self.tau2 = leaf_variance(k, m)
        self.m, self.k = m, k
        self.tree_prior, self.noise_prior, self.scale = tree_prior, noise_prior, scale

    def sweep(self, cache, sigma2, y_normalized, kernel, rng, *, fixed_sigma2=None, verify_cache=False,
              bounded_kernels=None):
        updates = []
        max_error = 0.0
        for j in range(self.m):
            residuals = cache.partial_residuals(y_normalized, j)
            active_kernel = kernel if bounded_kernels is None else bounded_kernels[j]
            tree, update = update_tree(cache.trees[j], sigma2, cache.batch, residuals,
                                       active_kernel, rng, self.tau2, self.tree_prior)
            cache.replace(j, tree)
            if bounded_kernels is not None:
                distance = tree.structure_distance(active_kernel.anchor)
                active_kernel.max_distance = max(active_kernel.max_distance, distance)
            updates.append(update)
            if verify_cache:
                max_error = max(max_error, cache.verify())
        residuals = y_normalized - cache.total
        # Shared sigma2 conditional is sampled only after the entire ensemble.
        sigma2 = self.noise_prior.sample(residuals, rng) if fixed_sigma2 is None else fixed_sigma2
        leaves = [path for tree in cache.trees for path in tree.partition(cache.batch)]
        diagnostic = EnsembleSweep(tuple(updates), float(sigma2), float(residuals @ residuals),
                                   len(leaves), max(map(len, leaves)))
        return float(sigma2), diagnostic, max_error

    def fit(self, batch, y, rng, *, burn_in=500, draws=1000, fixed_sigma2=None,
            verify_cache=False, initial_trees=None, max_structure_distance=None):
        """Run MCMC on fixed X/y, optionally initializing from a previous draw.

        fixed_sigma2 is in normalized units and is intended for numerical checks.
        initial_trees use normalized leaf values. Unsupported trees start from a stump.
        With a structure limit, support pruning consumes the same distance budget.
        Every sampled structure stays in that fixed neighbourhood of the inherited tree.
        verify_cache recomputes predictions after EACH tree update; leave it off
        for normal inference. No thinning or sampling of new minibatches occurs.
        """
        if type(burn_in) is not int or burn_in < 0 or type(draws) is not int or draws < 1:
            raise ValueError('Require nonnegative burn_in and positive draws')
        if max_structure_distance is not None:
            if type(max_structure_distance) is not int or max_structure_distance < 0:
                raise ValueError('Structure distance must be a nonnegative integer')
            if initial_trees is None:
                raise ValueError('A structure limit requires inherited trees')
        y = residual_vector(y).copy()
        if not len(y) or len(y) != len(batch):
            raise ValueError('fit requires nonempty aligned X and y')
        if fixed_sigma2 is not None:
            positive(fixed_sigma2, 'fixed_sigma2')
        started = time.perf_counter()
        batch = EncodedBatch(batch.values, batch.observed, batch.catalog, batch.catalog_hash)
        normalized = self.scale.normalize(y)
        if not np.isfinite(normalized).all():
            raise ValueError('Non-finite normalized responses')
        normalized.flags.writeable = False
        rules = RuleCatalog(RelationalEncoder(batch.catalog))
        kernel = (GrowPruneKernel if self.kernel == "grow_prune" else RevisionKernel)(batch, rules, self.tree_prior)
        data_hash = hashlib.sha256(batch.catalog_hash.encode() + batch.values.tobytes()
                                   + batch.observed.tobytes() + y.tobytes()).hexdigest()
        stump = Tree(float(normalized.mean() / self.m))
        trees = [stump] * self.m
        reused = None
        bounded_kernels = None
        repairs = None
        if initial_trees is not None:
            initial_trees = tuple(initial_trees)
            if len(initial_trees) != self.m or not all(isinstance(tree, Tree) for tree in initial_trees):
                raise ValueError('Expected one initial Tree per ensemble component')
            reused = 0
            for i, tree in enumerate(initial_trees):
                if max_structure_distance is not None:
                    trees[i] = repair_tree_support(tree, batch, rules, self.tree_prior)
                    distance = trees[i].structure_distance(tree)
                    if distance > max_structure_distance:
                        raise StructureLimitError(i, distance, max_structure_distance)
                    reused += 1
                elif np.isfinite(self.tree_prior.log_prior(tree, batch, rules)):
                    trees[i] = tree
                    reused += 1
            if max_structure_distance is not None:
                bounded_kernels = [BoundedStructureKernel(kernel, tree, max_structure_distance)
                                   for tree in initial_trees]
                repairs = tuple(tree.structure_distance(anchor) for tree, anchor in zip(trees, initial_trees))
                for bounded, distance in zip(bounded_kernels, repairs):
                    bounded.max_distance = distance
        cache = PredictionCache(trees, batch)
        sigma2 = self.noise_prior.lambda_ if fixed_sigma2 is None else fixed_sigma2
        retained, trace = [], []
        max_cache_error = 0.0 if verify_cache else None
        for sweep in range(burn_in + draws):
            sigma2, diagnostic, error = self.sweep(cache, sigma2, normalized, kernel, rng,
                                                    fixed_sigma2=fixed_sigma2, verify_cache=verify_cache,
                                                    bounded_kernels=bounded_kernels)
            trace.append(diagnostic)
            if verify_cache:
                max_cache_error = max(max_cache_error, error)
            if sweep >= burn_in:
                retained.append(EnsembleDraw(tuple(cache.trees), sigma2))
        posterior = BARTPosterior(tuple(retained), batch.catalog_hash, data_hash, self.scale,
                             self.tree_prior, self.noise_prior, self.k, burn_in, fixed_sigma2,
                             tuple(trace), int(np.count_nonzero((y < self.scale.lower) | (y > self.scale.upper))),
                             time.perf_counter() - started, max_cache_error, reused)
        if bounded_kernels is not None:
            posterior = replace(posterior, structure_limit=max_structure_distance,
                structure_distances=tuple(k.max_distance for k in bounded_kernels),
                support_repairs=repairs, distance_rejections=sum(k.distance_rejections for k in bounded_kernels))
        return posterior
