"""Reversible grow/prune proposals with fixed 1/2 move probabilities."""

from dataclasses import dataclass
import math

import numpy as np

from .tree import Tree


@dataclass(frozen=True)
class Proposal:
    candidate: Tree
    log_q_forward: float
    log_q_reverse: float
    move: str
    path: tuple[bool, ...]
    impossible: bool = False


class GrowPruneKernel:
    def __init__(self, batch, rules, prior):
        self.batch, self.rules, self.prior = batch, rules, prior

    def choices(self, tree):
        grow, prune = {}, {}
        for path, node, rows in tree.walk(self.batch):
            if node.is_leaf:
                eligible = self.prior.eligible(self.rules, self.batch, rows, len(path))
                if eligible:
                    grow[path] = eligible
            elif node.true_child.is_leaf and node.false_child.is_leaf:
                prune[path] = node.rule
        return grow, prune

    def grow_at(self, tree, path, rule):
        grow, _ = self.choices(tree)
        if path not in grow or rule not in grow[path].get(rule.relation, ()):
            raise ValueError("Ineligible grow proposal")
        candidate = tree.replace(
            path, Tree(rule=rule, true_child=Tree(), false_child=Tree())
        )
        _, reverse_prune = self.choices(candidate)
        forward = (
            math.log(0.5)
            - math.log(len(grow))
            + self.rules.log_probability(rule, grow[path])
        )
        reverse = math.log(0.5) - math.log(len(reverse_prune))
        return Proposal(candidate, forward, reverse, "grow", path)

    def prune_at(self, tree, path):
        _, prune = self.choices(tree)
        if path not in prune:
            raise ValueError("Ineligible prune proposal")
        candidate = tree.replace(path, Tree())
        reverse_grow, _ = self.choices(candidate)
        forward = math.log(0.5) - math.log(len(prune))
        reverse = (
            math.log(0.5)
            - math.log(len(reverse_grow))
            + self.rules.log_probability(prune[path], reverse_grow[path])
        )
        return Proposal(candidate, forward, reverse, "prune", path)

    @staticmethod
    def _impossible(tree, move):
        # An individual move-selection event. When BOTH moves are impossible,
        # these two events add to probability one on the same structure.
        return Proposal(tree, math.log(0.5), math.log(0.5), move, (), True)

    def propose(self, tree, rng):
        grow, prune = self.choices(tree)
        move = "grow" if rng.random() < 0.5 else "prune"
        choices = grow if move == "grow" else prune
        if not choices:
            return self._impossible(tree, move)
        path = tuple(choices)[int(rng.integers(len(choices)))]
        if move == "prune":
            return self.prune_at(tree, path)
        groups = grow[path]
        relation = tuple(groups)[int(rng.integers(len(groups)))]
        rules = groups[relation]
        return self.grow_at(tree, path, rules[int(rng.integers(len(rules)))])

    def all_proposals(self, tree):
        """Enumerate proposal events for tiny exact checks, not normal sampling."""
        grow, prune = self.choices(tree)
        if not grow:
            yield self._impossible(tree, "grow")
        for path, groups in grow.items():
            for rules in groups.values():
                for rule in rules:
                    yield self.grow_at(tree, path, rule)
        if not prune:
            yield self._impossible(tree, "prune")
        for path in prune:
            yield self.prune_at(tree, path)


def log_acceptance_ratio(tree, proposal, batch, residuals, sigma2, tau2, prior, rules):
    from .likelihood import tree_log_marginal

    if proposal.impossible:
        return 0.0
    candidate = proposal.candidate
    value = (
        tree_log_marginal(candidate, batch, residuals, sigma2, tau2)
        - tree_log_marginal(tree, batch, residuals, sigma2, tau2)
        + prior.log_prior(candidate, batch, rules)
        - prior.log_prior(tree, batch, rules)
        + proposal.log_q_reverse
        - proposal.log_q_forward
    )
    if np.isnan(value):
        raise FloatingPointError("Undefined MH acceptance ratio")
    return float(value)


class RevisionKernel(GrowPruneKernel):
    """Equal mixture of grow/prune/change/swap; invalid revisions are self loops.

    Change selects uniformly an internal node, then a relation/rule eligible at
    that node (including its current rule). Descendant support is checked after
    proposing, without conditioning the proposal on validity. Swap selects an
    internal parent-child edge uniformly and exchanges their rules. The edge
    set and topology do not change, so swap is its own inverse.
    """

    def _revision(self, tree, candidate, move, path, forward, reverse):
        impossible = not math.isfinite(
            self.prior.log_prior(candidate, self.batch, self.rules)
        )
        return Proposal(
            tree if impossible else candidate, forward, reverse, move, path, impossible
        )

    def change_events(self, tree):
        nodes = [(p, n, rows) for p, n, rows in tree.walk(self.batch) if not n.is_leaf]
        if not nodes:
            yield Proposal(tree, math.log(0.25), math.log(0.25), "change", (), True)
        for path, node, rows in nodes:
            groups = self.prior.eligible(self.rules, self.batch, rows, len(path))
            base = math.log(0.25) - math.log(len(nodes))
            for rules in groups.values():
                for rule in rules:
                    candidate = tree.replace(
                        path,
                        Tree(
                            rule=rule,
                            true_child=node.true_child,
                            false_child=node.false_child,
                        ),
                    )
                    yield self._revision(
                        tree,
                        candidate,
                        "change",
                        path,
                        base + self.rules.log_probability(rule, groups),
                        base + self.rules.log_probability(node.rule, groups),
                    )

    def swap_events(self, tree):
        nodes = {p: n for p, n, _ in tree.walk(self.batch) if not n.is_leaf}
        edges = [p for p in nodes if p]
        if not edges:
            yield Proposal(tree, math.log(0.25), math.log(0.25), "swap", (), True)
        for path in edges:
            parent, child = nodes[path[:-1]], nodes[path]
            revised_child = Tree(
                rule=parent.rule,
                true_child=child.true_child,
                false_child=child.false_child,
            )
            revised_parent = Tree(
                rule=child.rule,
                true_child=parent.true_child,
                false_child=parent.false_child,
            )
            revised_parent = revised_parent.replace((path[-1],), revised_child)
            candidate = tree.replace(path[:-1], revised_parent)
            probability = math.log(0.25) - math.log(len(edges))
            yield self._revision(
                tree, candidate, "swap", path, probability, probability
            )

    def all_proposals(self, tree):
        from dataclasses import replace

        for proposal in super().all_proposals(tree):
            yield replace(
                proposal,
                log_q_forward=proposal.log_q_forward - math.log(2),
                log_q_reverse=proposal.log_q_reverse - math.log(2),
            )
        yield from self.change_events(tree)
        yield from self.swap_events(tree)

    def propose(self, tree, rng):
        from dataclasses import replace

        move = ("grow", "prune", "change", "swap")[int(rng.integers(4))]
        if move in ("grow", "prune"):
            grow, prune = self.choices(tree)
            choices = grow if move == "grow" else prune
            if not choices:
                return Proposal(tree, math.log(0.25), math.log(0.25), move, (), True)
            path = tuple(choices)[int(rng.integers(len(choices)))]
            if move == "prune":
                proposal = self.prune_at(tree, path)
            else:
                groups = grow[path]
                group = groups[tuple(groups)[int(rng.integers(len(groups)))]]
                proposal = self.grow_at(
                    tree, path, group[int(rng.integers(len(group)))]
                )
            return replace(
                proposal,
                log_q_forward=proposal.log_q_forward - math.log(2),
                log_q_reverse=proposal.log_q_reverse - math.log(2),
            )
        nodes = [(p, n, rows) for p, n, rows in tree.walk(self.batch) if not n.is_leaf]
        choices = nodes if move == "change" else [item for item in nodes if item[0]]
        if not choices:
            return Proposal(tree, math.log(0.25), math.log(0.25), move, (), True)
        path, node, rows = choices[int(rng.integers(len(choices)))]
        base = math.log(0.25) - math.log(len(choices))
        if move == "change":
            groups = self.prior.eligible(self.rules, self.batch, rows, len(path))
            group = groups[tuple(groups)[int(rng.integers(len(groups)))]]
            rule = group[int(rng.integers(len(group)))]
            candidate = tree.replace(
                path,
                Tree(
                    rule=rule, true_child=node.true_child, false_child=node.false_child
                ),
            )
            return self._revision(
                tree,
                candidate,
                move,
                path,
                base + self.rules.log_probability(rule, groups),
                base + self.rules.log_probability(node.rule, groups),
            )
        parent = next(n for p, n, _ in nodes if p == path[:-1])
        child = Tree(
            rule=parent.rule, true_child=node.true_child, false_child=node.false_child
        )
        revised = Tree(
            rule=node.rule, true_child=parent.true_child, false_child=parent.false_child
        )
        candidate = tree.replace(path[:-1], revised.replace((path[-1],), child))
        return self._revision(tree, candidate, move, path, base, base)
