"""Frequentist binary controls on the same categorical rules and frozen targets.

Greedy CART minimizes training SSE; no F-test, posterior, or uncertainty claim.
Incremental control uses TD updates and refines the visited leaf using its FIFO
TD-target buffer. Both are declared experimental controls, not legacy replicas.
"""

from dataclasses import dataclass
import time

import numpy as np

from .agent import BayesianQAgent, BayesianQModel
from .encoding import RelationalEncoder
from .rules import RuleCatalog
from .targets import q_learning_update
from .tree import Tree


def fit_cart(batch, y, prior, *, max_depth=None):
    y = np.asarray(y, dtype=float)
    rules = RuleCatalog(RelationalEncoder(batch.catalog))
    limit = prior.max_depth if max_depth is None else max_depth

    def build(rows, depth):
        leaf = Tree(float(y[rows].mean()))
        if depth >= limit:
            return leaf
        groups = rules.eligible(
            batch, rows, depth=depth, max_depth=limit, min_child=prior.min_child
        )
        best, gain = None, 1e-12
        total = float(np.sum((y[rows] - y[rows].mean()) ** 2))
        for choices in groups.values():
            for rule in choices:
                route = rule.route(batch)[rows]
                a, b = rows[route], rows[~route]
                improvement = (
                    total
                    - np.sum((y[a] - y[a].mean()) ** 2)
                    - np.sum((y[b] - y[b].mean()) ** 2)
                )
                if improvement > gain:
                    best, gain = (rule, a, b), improvement
        if best is None:
            return leaf
        rule, a, b = best
        return Tree(
            rule=rule, true_child=build(a, depth + 1), false_child=build(b, depth + 1)
        )

    if not len(y) or len(batch) != len(y) or not np.isfinite(y).all():
        raise ValueError("CART needs nonempty aligned finite targets")
    return build(np.arange(len(y)), 0)


@dataclass(frozen=True)
class PointTreeModel:
    tree: Tree
    catalog_hash: str
    scale: object
    targets_outside_scale: int = 0
    elapsed_seconds: float = 0.0

    def predict_mean(self, batch):
        if batch.catalog_hash != self.catalog_hash:
            raise ValueError("Catalog mismatch")
        return self.tree.predict(batch)

    def predict_latent_draws(self, batch):
        raise ValueError("Frequentist control does not define posterior draws")

    def diagnostics(self):
        return {
            "estimator": "greedy_binary_SSE",
            "elapsed_seconds": self.elapsed_seconds,
            "targets_outside_scale": self.targets_outside_scale,
        }


class CARTRegressor:
    def __init__(self, config):
        self.config = config

    def fit(self, batch, y, rng, **options):
        started = time.perf_counter()
        tree = fit_cart(batch, y, self.config.tree_prior)
        scale = self.config.scale
        return PointTreeModel(
            tree,
            batch.catalog_hash,
            scale,
            int(np.count_nonzero((y < scale.lower) | (y > scale.upper))),
            time.perf_counter() - started,
        )


class BatchTreeAgent(BayesianQAgent):
    def make_regressor(self, action, calibration):
        return CARTRegressor(self.config)


class IncrementalBinaryAgent(BayesianQAgent):
    def __init__(self, config, encoder, *, eta=0.025, split_min=32):
        super().__init__(config, encoder)
        if not 0 < eta <= 1 or split_min < 2:
            raise ValueError("Invalid incremental control parameters")
        self.eta, self.split_min = eta, split_min
        self.model = BayesianQModel(
            config.actions,
            tuple(
                PointTreeModel(Tree(), encoder.catalog_hash, config.scale)
                for _ in config.actions
            ),
            encoder.catalog_hash,
        )
        self.leaf_buffers = {}

    def observe(self, *args, **kwargs):
        transition = super().observe(*args, **kwargs)
        action, batch = transition.action, transition.state
        estimator = self.model.posteriors[action]
        path, (leaf, _) = next(
            (p, item)
            for p, item in estimator.tree.partition(batch).items()
            if len(item[1])
        )
        update = q_learning_update(
            leaf.mu,
            transition.reward,
            lambda: self.model.predict(transition.next_state).max(),
            eta_q=self.eta,
            gamma=self.config.gamma,
            terminated=transition.terminated,
            truncated=transition.truncated,
            bootstrap_on_truncation=self.config.bootstrap_on_truncation,
        )
        buffer = self.leaf_buffers.setdefault((action, path), [])
        buffer.append((batch, update.target))
        del buffer[: -self.config.batch_size_per_action]
        replacement = Tree(update.new_q)
        if len(buffer) >= self.split_min and len(path) < self.config.max_depth:
            from .replay import concatenate_states

            data = concatenate_states([s for s, _ in buffer], self.encoder)
            candidate = fit_cart(
                data, [v for _, v in buffer], self.config.tree_prior, max_depth=1
            )
            if not candidate.is_leaf:
                replacement = candidate
                del self.leaf_buffers[(action, path)]
        models = list(self.model.posteriors)
        models[action] = PointTreeModel(
            estimator.tree.replace(path, replacement),
            self.encoder.catalog_hash,
            self.config.scale,
        )
        self.model = BayesianQModel(
            self.config.actions,
            tuple(models),
            self.encoder.catalog_hash,
            self.interactions,
        )
        return transition

    def fit_if_due(self, *, force=False):
        return None
