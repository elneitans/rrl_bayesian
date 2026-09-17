"""Small finite-state reference: exhaustive structures and exact MH matrix.

Exponential enumeration is intentionally restricted by max_trees. Leaf values
are marginalized and sigma2 is FIXED: this is not an exact reference for the
joint unknown-variance chain or for an ensemble.
"""

import math

import numpy as np
from scipy.special import logsumexp

from .likelihood import tree_log_marginal
from .proposals import GrowPruneKernel, log_acceptance_ratio
from .tree import Tree


def enumerate_trees(batch, rules, prior, *, max_trees=10000):
    def visit(rows, depth):
        result = [Tree()]
        groups = prior.eligible(rules, batch, rows, depth)
        for group in groups.values():
            for rule in group:
                route = rule.route(batch)[rows]
                true_trees = visit(rows[route], depth + 1)
                false_trees = visit(rows[~route], depth + 1)
                if len(result) + len(true_trees) * len(false_trees) > max_trees:
                    raise ValueError("Exact enumeration exceeds max_trees; reduce depth/data")
                result.extend(Tree(rule=rule, true_child=t, false_child=f)
                              for t in true_trees for f in false_trees)
        return result
    return tuple(visit(np.arange(len(batch)), 0))


def exact_posterior(trees, batch, y_normalized, sigma2, tau2, prior, rules):
    logs = np.array([prior.log_prior(t, batch, rules)
                     + tree_log_marginal(t, batch, y_normalized, sigma2, tau2) for t in trees])
    return np.exp(logs - logsumexp(logs))


def transition_matrix(trees, batch, y_normalized, sigma2, tau2, prior, rules):
    index = {tree.structure_key(): i for i, tree in enumerate(trees)}
    if len(index) != len(trees):
        raise ValueError("Duplicate structural states")
    kernel = GrowPruneKernel(batch, rules, prior)
    matrix = np.zeros((len(trees), len(trees)))
    for i, tree in enumerate(trees):
        for proposal in kernel.all_proposals(tree):
            probability = math.exp(proposal.log_q_forward)
            if proposal.impossible:
                matrix[i, i] += probability
                continue
            j = index[proposal.candidate.structure_key()]
            log_ratio = log_acceptance_ratio(tree, proposal, batch, y_normalized, sigma2, tau2, prior, rules)
            acceptance = math.exp(min(0.0, log_ratio))
            matrix[i, j] += probability * acceptance
            matrix[i, i] += probability * (1 - acceptance)
    return matrix


def exact_latent_moments(trees, probabilities, batch, y_normalized, sigma2, tau2):
    from .likelihood import LeafStats
    means, seconds = [], []
    for tree in trees:
        mean, second = np.empty(len(batch)), np.empty(len(batch))
        for leaf, rows in tree.partition(batch).values():
            b, v = LeafStats.from_residuals(y_normalized[rows]).posterior(sigma2, tau2)
            mean[rows], second[rows] = b, v + b*b
        means.append(mean)
        seconds.append(second)
    mean = probabilities @ np.array(means)
    return mean, probabilities @ np.array(seconds) - mean*mean


def exact_ensemble_posterior(trees, m, batch, y_normalized, sigma2, tau2, prior, rules,
                             *, max_ensembles=1000):
    """Independent Gaussian reference integrating ALL leaf means jointly.

    For a fixed ensemble, f ~ Normal(0, K) with K=tau2*Z*Z.T where Z is the
    concatenated leaf-membership design. Enumerate ordered ensembles; this
    does not multiply single-tree marginal likelihoods, which would be wrong.
    """
    from itertools import product
    from scipy.linalg import cho_factor, cho_solve
    from scipy.stats import multivariate_normal
    from .likelihood import positive, residual_vector
    positive(sigma2, 'sigma2')
    positive(tau2, 'tau2')
    y = residual_vector(y_normalized)
    if type(m) is not int or m < 1 or len(trees)**m > max_ensembles:
        raise ValueError('Invalid m or exact ensemble enumeration exceeds limit')
    if len(y) != len(batch) or not len(y):
        raise ValueError('Need nonempty aligned batch and responses')
    covariance_by_tree, priors = [], []
    for tree in trees:
        membership = np.zeros((len(batch), len(tree.partition(batch))))
        for j, (leaf, rows) in enumerate(tree.partition(batch).values()):
            membership[rows, j] = 1
        covariance_by_tree.append(tau2 * membership @ membership.T)
        priors.append(prior.log_prior(tree, batch, rules))
    indices = list(product(range(len(trees)), repeat=m))
    logs, means, covariances = [], [], []
    for choices in indices:
        latent_cov = sum(covariance_by_tree[j] for j in choices)
        observation_cov = latent_cov + sigma2 * np.eye(len(batch))
        factor = cho_factor(observation_cov, lower=True)
        logs.append(sum(priors[j] for j in choices) + multivariate_normal.logpdf(y, cov=observation_cov))
        mean = latent_cov @ cho_solve(factor, y)
        conditional_cov = latent_cov - latent_cov @ cho_solve(factor, latent_cov)
        means.append(mean)
        covariances.append(conditional_cov)
    probabilities = np.exp(logs - logsumexp(logs))
    mean = probabilities @ np.array(means)
    second = sum(p * (cov + np.outer(mu, mu)) for p, cov, mu in zip(probabilities, covariances, means))
    ensembles = tuple(tuple(trees[j] for j in choices) for choices in indices)
    return ensembles, probabilities, mean, second - np.outer(mean, mean)
