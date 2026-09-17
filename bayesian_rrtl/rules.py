"""Binary predicates and fixed rule catalog, grouped by semantic relation."""

from dataclasses import dataclass
import math

import numpy as np

from .encoding import CATEGORIES, RelationKey


@dataclass(frozen=True)
class Rule:
    relation: RelationKey
    operator: str = "equals"
    value: str | None = None

    def __post_init__(self):
        if self.operator not in ("equals", "is_missing"):
            raise ValueError("Only equals and is_missing rules are supported")
        if (
            self.operator == "equals"
            and self.value not in CATEGORIES[self.relation.type]
        ):
            raise ValueError("Invalid rule category")
        if self.operator == "is_missing" and self.value is not None:
            raise ValueError("is_missing does not take a category")

    def route(self, batch):
        """True goes to the true child; missing ordinary tests return False."""
        try:
            j = batch.catalog.index(self.relation)
        except ValueError as exc:
            raise ValueError("Rule relation absent from batch catalog") from exc
        if self.operator == "is_missing":
            return ~batch.observed[:, j]
        code = CATEGORIES[self.relation.type].index(self.value)
        return batch.observed[:, j] & (batch.values[:, j] == code)


class RuleCatalog:
    def __init__(self, encoder):
        self.catalog_hash = encoder.catalog_hash
        self.rules = tuple(
            rule
            for key in encoder.catalog
            for rule in (
                *[Rule(key, "equals", v) for v in CATEGORIES[key.type]],
                Rule(key, "is_missing"),
            )
        )

    def eligible(self, batch, indices=None, *, depth=0, max_depth=6, min_child=1):
        """Indices must be the observations reaching this node through ancestors.

        Keep rule identity even when two predicates induce the same partition.
        Return eligible rules grouped by relation for hierarchical probabilities.
        """
        if batch.catalog_hash != self.catalog_hash:
            raise ValueError("Rule catalog and batch differ")
        if min_child < 1 or depth < 0 or max_depth < 0:
            raise ValueError("Invalid tree support limits")
        if depth >= max_depth:
            return {}
        # Count nominal categories directly by column. Reconstructing EncodedBatch
        # and linearly searching its catalog per rule dominated large Atari catalogs.
        values = batch.values if indices is None else batch.values[indices]
        observed = batch.observed if indices is None else batch.observed[indices]
        groups = {}
        offset = 0
        for j, key in enumerate(batch.catalog):
            categories = CATEGORIES[key.type]
            counts = [
                int(np.count_nonzero(observed[:, j] & (values[:, j] == code)))
                for code in range(len(categories))
            ]
            counts.append(int(np.count_nonzero(~observed[:, j])))
            for rule, n_true in zip(self.rules[offset : offset + len(counts)], counts):
                if min(n_true, len(values) - n_true) >= min_child:
                    groups.setdefault(rule.relation, []).append(rule)
            offset += len(counts)
        return {key: tuple(rules) for key, rules in groups.items()}

    @staticmethod
    def log_probability(rule, eligible):
        group = eligible.get(rule.relation, ())
        if rule not in group:
            return -math.inf
        return -math.log(len(eligible)) - math.log(len(group))
