"""Experimental growth criterion over the historical updated-Q statistics.

These p-values are descriptive growth scores, not calibrated false-positive
guarantees under temporal dependence, candidate selection and repeated tests.
"""
from dataclasses import dataclass
import math
import sys

from scipy.stats import f

from .baseline import CorrectedNode


@dataclass(frozen=True)
class ANOVAResult:
    n: int
    k: int
    within: float
    between: float
    statistic: float | None
    p_value: float | None
    reason: str | None = None


def anova_from_stats(groups):
    """One-way ANOVA from n, mean and Welford J, with consistency checks.

    Empty groups do not contribute degrees of freedom. Tiny discrepancies in
    independently accumulated means/SS are allowed, but negative SS, unequal
    counts and materially inconsistent pooled moments are errors, not no-splits.
    """
    if 'Total' not in groups or len(groups) < 2:
        raise ValueError('ANOVA requires Total and category statistics')
    for stats in groups.values():
        if (set(stats) != {'n', 'mu', 'J'} or type(stats['n']) is not int or stats['n'] < 0
                or not math.isfinite(stats['mu']) or not math.isfinite(stats['J'])
                or stats['J'] < 0 or (stats['n'] == 0 and stats['J'] != 0)):
            raise ValueError('Invalid ANOVA count/mean/sum of squares')
    total = groups['Total']
    nonempty = [s for name, s in groups.items() if name != 'Total' and s['n'] > 0]
    n, k = sum(s['n'] for s in nonempty), len(nonempty)
    if n != total['n']:
        raise ValueError('Inconsistent ANOVA category/total counts')
    try:
        within = math.fsum(s['J'] for s in nonempty)
        between = math.fsum(s['n'] * (s['mu'] - total['mu'])**2 for s in nonempty)
        pooled_mean = math.fsum((s['n']/n)*s['mu'] for s in nonempty) if n else 0.
    except OverflowError as exc:
        raise ValueError('ANOVA moments exceed numeric range') from exc
    if not all(math.isfinite(v) for v in (within, between, pooled_mean, within+between)):
        raise ValueError('Nonfinite ANOVA moments')
    mean_scale = max([1., *(abs(s['mu']) for s in nonempty)])
    if (not math.isclose(total['mu'], pooled_mean, rel_tol=1e-8, abs_tol=1e-10*mean_scale)
            or (total['J'] == 0) != (within+between == 0)
            or not math.isclose(total['J'], within+between, rel_tol=1e-8, abs_tol=1e-12)):
        raise ValueError('Inconsistent ANOVA pooled mean/sum of squares')
    if k < 2:
        return ANOVAResult(n, k, within, between, None, None, 'fewer_than_two_groups')
    if n <= k:
        return ANOVAResult(n, k, within, between, None, None, 'no_residual_degrees_of_freedom')
    if within+between == 0:
        return ANOVAResult(n, k, within, between, None, None, 'zero_total_dispersion')
    if within == 0:
        return ANOVAResult(n, k, within, between, math.inf, 0.)
    if between == 0:
        statistic = 0.
    else:
        # Avoid underflow in W/(n-K) or an overflowing intermediate B/W.
        log_f = math.log(between)+math.log(n-k)-math.log(within)-math.log(k-1)
        statistic = math.inf if log_f > math.log(sys.float_info.max) else math.exp(log_f)
    p_value = float(f.sf(statistic, k-1, n-k))
    if not math.isfinite(p_value) or not 0 <= p_value <= 1:
        raise FloatingPointError('ANOVA survival function returned an invalid p-value')
    return ANOVAResult(n, k, within, between, statistic, p_value)


class ANOVANode(CorrectedNode):
    """Same routing, Q statistics and branches; a different growth test only."""

    def anova_test(self, literal):
        return anova_from_stats(self.refinements_stats[literal])

    def find_best_literal(self):
        best, best_p = None, math.inf
        for literal in self.refinements:
            if self.refinements_stats[literal]['Total']['n'] <= self.min_sample_size:
                continue  # Preserve the historical strict minimum (n > minimum).
            result = self.anova_test(literal)
            if result.p_value is not None and result.p_value < best_p:
                best, best_p = literal, result.p_value  # Stable catalogue-order ties.
        return best

    def split_iteration(self, state, inherit_q_values):
        if self.root_action not in self.valid_actions or not any(f.obj2 is not None for f in state):
            return False
        node = self.get_state_leaf(state)
        if node.depth >= self.max_depth or node.literal is not None:
            return False
        literal = node.find_best_literal()
        if literal is None or node.anova_test(literal).p_value > node.significance_level:
            return False
        remaining = [candidate for candidate in node.refinements if candidate != literal]
        if literal.type == 'comparative':
            labels = ('more', 'same', 'less')
        elif literal.type == 'logical':
            labels = ('True', 'False')
        else:
            raise ValueError('Unrecognized ANOVA literal type')
        children = []
        for label in labels:
            child = ANOVANode(root_action=node.root_action, refinements=remaining,
                max_depth=node.max_depth, min_sample_size=node.min_sample_size,
                node_type=label, parent=node, depth=node.depth+1,
                conjuntion_str=node.conjuntion_str+str(literal)+label,
                significance_level=node.significance_level,
                best_literal_criteria=node.best_literal_criteria, valid_actions=node.valid_actions)
            if inherit_q_values:
                child.q_value = node.q_value
            setattr(node, label.lower()+'_branch', child)
            children.append(child)
        node.children = children
        node.literal = literal
        return True
