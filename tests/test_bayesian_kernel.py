from collections import namedtuple
import math

import numpy as np
import pytest

from bayesian_rrtl import RelationalEncoder, Tree
from bayesian_rrtl.exact import enumerate_trees, exact_posterior, transition_matrix
from bayesian_rrtl.priors import TreePrior
from bayesian_rrtl.proposals import GrowPruneKernel, Proposal, log_acceptance_ratio
from bayesian_rrtl.rules import RuleCatalog
from bayesian_rrtl.sampler import SingleTreeSampler

Fact = namedtuple('Fact', 'type value name obj1 obj2 obj3', defaults=[None, None])


def tiny_data():
    states = [[Fact('comparative', value, 'x', 'p_t', 'b_t'),
               Fact('logical', logical, 'present', 'b_t')]
              for value, logical in [('less', 'False'), ('same', 'True'), ('more', 'True')]]
    encoder = RelationalEncoder.from_states(states)
    return encoder.transform(states), np.array([-0.25, 0.1, 0.3]), RuleCatalog(encoder)


@pytest.mark.parametrize('depth,min_child', [(0, 1), (1, 1), (2, 1), (2, 2)])
def test_exact_prior_and_detailed_balance(depth, min_child):
    batch, y, rules = tiny_data()
    prior = TreePrior(alpha_tree=0.6, beta_tree=1, max_depth=depth, min_child=min_child)
    trees = enumerate_trees(batch, rules, prior)
    assert sum(math.exp(prior.log_prior(t, batch, rules)) for t in trees) == pytest.approx(1)
    probability = exact_posterior(trees, batch, y, 0.1, 0.0625, prior, rules)
    matrix = transition_matrix(trees, batch, y, 0.1, 0.0625, prior, rules)
    np.testing.assert_allclose(matrix.sum(axis=1), 1, atol=1e-14)
    assert np.all(matrix >= 0)
    flow = probability[:, None] * matrix
    np.testing.assert_allclose(flow, flow.T, atol=1e-14)
    np.testing.assert_allclose(probability @ matrix, probability, atol=1e-14)
    if depth == 0:
        np.testing.assert_allclose(matrix, [[1]])


def test_proposal_inverses_and_original_tree_immutable():
    batch, y, rules = tiny_data()
    prior = TreePrior(max_depth=2)
    kernel = GrowPruneKernel(batch, rules, prior)
    trees = enumerate_trees(batch, rules, prior)
    for tree in trees:
        before = tree.structure_key()
        proposals = list(kernel.all_proposals(tree))
        assert sum(math.exp(p.log_q_forward) for p in proposals) == pytest.approx(1)
        for proposal in proposals:
            if proposal.impossible:
                continue
            if proposal.move == 'grow':
                reverse = kernel.prune_at(proposal.candidate, proposal.path)
            else:
                old_node = next(n for path, n, rows in tree.walk(batch) if path == proposal.path)
                reverse = kernel.grow_at(proposal.candidate, proposal.path, old_node.rule)
            assert reverse.candidate.structure_key() == before
            assert proposal.log_q_forward == pytest.approx(reverse.log_q_reverse)
            assert proposal.log_q_reverse == pytest.approx(reverse.log_q_forward)
            lr = log_acceptance_ratio(tree, proposal, batch, y, 0.1, 0.0625, prior, rules)
            rev_lr = log_acceptance_ratio(proposal.candidate, reverse, batch, y, 0.1, 0.0625, prior, rules)
            assert lr == pytest.approx(-rev_lr)
        assert tree.structure_key() == before


def test_missing_support_and_impossible_events():
    encoder = RelationalEncoder.from_states([[Fact('logical', 'False', 'present', 'ball_t')], []])
    batch = encoder.transform([[Fact('logical', 'False', 'present', 'ball_t')], []])
    rules, prior = RuleCatalog(encoder), TreePrior(max_depth=1)
    trees = enumerate_trees(batch, rules, prior)
    assert len(trees) == 3  # False vs missing are different named predicates.
    matrix = transition_matrix(trees, batch, np.array([0.1, 0.2]), 0.2, 0.1, prior, rules)
    np.testing.assert_allclose(matrix.sum(axis=1), 1)
    prior_zero = TreePrior(max_depth=0)
    invalid = trees[1]
    assert prior_zero.log_prior(invalid, batch, rules) == -math.inf
    proposals = list(GrowPruneKernel(batch, rules, prior_zero).all_proposals(Tree()))
    assert len(proposals) == 2 and all(p.impossible for p in proposals)


@pytest.mark.parametrize('impossible', [True, False])
def test_leaves_resampled_even_after_impossible_or_rejected_move(impossible):
    batch, y, rules = tiny_data()
    prior = TreePrior(max_depth=0)
    sampler = SingleTreeSampler(tree_prior=prior)
    tree = Tree(99)
    class ForcedKernel:
        def __init__(self):
            self.rules = rules
        def propose(self, current, rng):
            candidate = Tree(rule=rules.rules[0], true_child=Tree(), false_child=Tree())
            return Proposal(current if impossible else candidate, 0, 0, 'grow', (), impossible)
    sampled, sigma2, diagnostic = sampler.sweep(tree, 0.1, batch, y, ForcedKernel(),
                                               np.random.default_rng(9), fixed_sigma2=0.1)
    assert not diagnostic.accepted
    assert diagnostic.impossible == impossible
    assert sampled.structure_key() == tree.structure_key()
    assert sampled.mu != tree.mu
    assert tree.mu == 99
    assert sigma2 == 0.1


def test_enumeration_has_an_explicit_size_guard():
    batch, y, rules = tiny_data()
    with pytest.raises(ValueError, match='max_trees'):
        enumerate_trees(batch, rules, TreePrior(max_depth=2), max_trees=2)


def test_hand_computed_prior_and_grow_probability():
    batch, y, rules = tiny_data()
    prior = TreePrior(alpha_tree=0.6, beta_tree=1, max_depth=2)
    rule = next(r for r in rules.rules if r.relation.type == 'comparative' and r.value == 'less')
    proposal = GrowPruneKernel(batch, rules, prior).grow_at(Tree(), (), rule)
    assert math.exp(prior.log_prior(Tree(), batch, rules)) == pytest.approx(0.4)
    # Root: .6 * (1/2 relations) * (1/3 categories); singleton cannot split.
    # Remaining child can split with probability .6/(1+1)=.3.
    assert math.exp(prior.log_prior(proposal.candidate, batch, rules)) == pytest.approx(0.07)
    assert math.exp(proposal.log_q_forward) == pytest.approx(0.5 / 2 / 3)
    assert math.exp(proposal.log_q_reverse) == pytest.approx(0.5)
