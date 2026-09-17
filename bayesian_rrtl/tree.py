"""Immutable binary trees with total routing; no learning or Atari dependency."""

from dataclasses import dataclass
import math

import numpy as np

from .rules import Rule


@dataclass(frozen=True)
class Tree:
    mu: float = 0.0
    rule: Rule | None = None
    true_child: "Tree | None" = None
    false_child: "Tree | None" = None

    def __post_init__(self):
        if not math.isfinite(self.mu):
            raise ValueError("Leaf contribution must be finite")
        children = (self.true_child, self.false_child)
        if self.rule is None and any(child is not None for child in children):
            raise ValueError("A leaf cannot have children")
        if self.rule is not None and not all(isinstance(child, Tree) for child in children):
            raise ValueError("A split must have exactly two Tree children")

    @property
    def is_leaf(self):
        return self.rule is None

    def structure_key(self):
        """Hashable identity excluding leaf parameters (collapsed MH state)."""
        if self.is_leaf:
            return ()
        return (self.rule, self.true_child.structure_key(), self.false_child.structure_key())

    def replace(self, path, subtree):
        """Return a new tree, leaving this tree and all posterior draws untouched."""
        if not path:
            return subtree
        if self.is_leaf:
            raise ValueError("Path descends beyond a leaf")
        if path[0]:
            return Tree(rule=self.rule, true_child=self.true_child.replace(path[1:], subtree),
                        false_child=self.false_child)
        return Tree(rule=self.rule, true_child=self.true_child,
                    false_child=self.false_child.replace(path[1:], subtree))

    def walk(self, batch):
        """Yield path, node and observations at every node, in stable order."""
        def visit(node, rows, path):
            yield path, node, rows
            if not node.is_leaf:
                mask = node.rule.route(batch)[rows]
                yield from visit(node.true_child, rows[mask], path + (True,))
                yield from visit(node.false_child, rows[~mask], path + (False,))
        yield from visit(self, np.arange(len(batch)), ())

    def partition(self, batch):
        """Return {path: (leaf, row_indices)}, including leaves with no rows."""
        result = {}

        def visit(node, indices, path):
            if node.is_leaf:
                result[path] = (node, indices)
                return
            mask = node.rule.route(batch)[indices]
            visit(node.true_child, indices[mask], path + (True,))
            visit(node.false_child, indices[~mask], path + (False,))

        visit(self, np.arange(len(batch)), ())
        return result

    def predict(self, batch):
        prediction = np.empty(len(batch), dtype=float)
        for leaf, indices in self.partition(batch).values():
            prediction[indices] = leaf.mu
        return prediction

    def validate_support(self, batch, rules, *, max_depth=6, min_child=1):
        """Check every split against support induced by its ancestors and X."""
        def visit(node, indices, depth):
            if node.is_leaf:
                return
            eligible = rules.eligible(batch, indices, depth=depth,
                                      max_depth=max_depth, min_child=min_child)
            if node.rule not in eligible.get(node.rule.relation, ()):
                raise ValueError("Split outside eligible support")
            mask = node.rule.route(batch)[indices]
            visit(node.true_child, indices[mask], depth + 1)
            visit(node.false_child, indices[~mask], depth + 1)

        visit(self, np.arange(len(batch)), 0)
