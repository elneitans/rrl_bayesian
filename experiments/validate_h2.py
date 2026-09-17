"""Run exact H2 validation: python -m experiments.validate_h2 --output <directory>."""

import argparse
from collections import namedtuple
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import scipy

from bayesian_rrtl import RelationalEncoder, RuleCatalog
from bayesian_rrtl.exact import (enumerate_trees, exact_latent_moments,
                                exact_posterior, transition_matrix)
from bayesian_rrtl.priors import TreePrior
from bayesian_rrtl.sampler import SingleTreeSampler

Fact = namedtuple('Fact', 'type value name obj1 obj2 obj3', defaults=[None, None])


def validate(output, seeds, draws, burn_in):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    states = [[Fact('comparative', category, 'x', 'player_t', 'ball_t'),
               Fact('logical', present, 'present', 'ball_t')]
              for category, present in [('less', 'False'), ('same', 'True'), ('more', 'True')]]
    encoder = RelationalEncoder.from_states(states)
    batch = encoder.transform(states)
    y = np.array([-0.25, 0.1, 0.3])
    prior = TreePrior(alpha_tree=0.6, beta_tree=1, max_depth=2)
    sampler = SingleTreeSampler(tree_prior=prior)
    rules = RuleCatalog(encoder)
    sigma2 = 0.1
    trees = enumerate_trees(batch, rules, prior)
    probability = exact_posterior(trees, batch, y, sigma2, sampler.tau2, prior, rules)
    matrix = transition_matrix(trees, batch, y, sigma2, sampler.tau2, prior, rules)
    mean, variance = exact_latent_moments(trees, probability, batch, y, sigma2, sampler.tau2)
    flow = probability[:, None] * matrix
    row_error = float(np.max(np.abs(matrix.sum(axis=1) - 1)))
    balance_error = float(np.max(np.abs(flow - flow.T)))
    stationary_error = float(np.max(np.abs(probability @ matrix - probability)))
    assert max(row_error, balance_error, stationary_error) < 1e-12
    index = {tree.structure_key(): i for i, tree in enumerate(trees)}
    symmetric = np.sqrt(probability[:, None]) * matrix / np.sqrt(probability[None, :])
    rho = float(np.max(np.abs(np.linalg.eigvalsh(symmetric)[:-1])))
    mc_bound = np.sqrt(probability * (1 - probability) * (1 + rho) / (1 - rho) / draws)
    chains, frequencies = [], []
    for seed in seeds:
        posterior = sampler.fit(batch, y, np.random.default_rng(seed), burn_in=burn_in,
                                draws=draws, fixed_sigma2=sigma2)
        frequency = np.bincount([index[t.structure_key()] for t in posterior.trees],
                                minlength=len(trees)) / draws
        error = np.abs(frequency - probability)
        tv = float(error.sum() / 2)
        assert np.all(error < 6 * mc_bound + 1 / draws), 'Posterior frequency exceeds MC tolerance'
        assert tv < 0.07, 'Total variation exceeds validation threshold'
        latent = posterior.predict_latent_draws(batch)
        mean_error = np.abs(latent.mean(axis=0) - mean)
        # The structural spectral bound also bounds the observable conditional
        # means; the remaining independently refreshed leaf noise is uncorrelated.
        mean_bound = np.sqrt(variance * (1 + rho) / (1 - rho) / draws)
        assert np.all(mean_error < 6 * mean_bound), 'Latent mean exceeds MC tolerance'
        chains.append({'seed': seed, 'total_variation': tv,
                       'maximum_frequency_error': float(error.max()),
                       'maximum_latent_mean_error': float(mean_error.max()),
                       'diagnostics': posterior.diagnostics()})
        frequencies.append(frequency)
    # Additional smoke check: unknown variance and a recoverable categorical signal.
    training_states = [states[i % 3] for i in range(90)]
    train = encoder.transform(training_states)
    signal = np.tile([-0.3, 0.0, 0.3], 30)
    response = signal + np.random.default_rng(712).normal(0, 0.04, len(signal))
    learned = SingleTreeSampler(tree_prior=TreePrior(max_depth=2)).fit(
        train, response, np.random.default_rng(713), burn_in=500, draws=1500)
    rmse = float(np.sqrt(np.mean((learned.predict_mean(train) - signal)**2)))
    assert rmse < 0.08 and np.isfinite(learned.sigma2).all()
    root = Path(__file__).resolve().parents[1]
    source_paths = [*sorted((root / 'bayesian_rrtl').glob('*.py')), Path(__file__).resolve()]
    report = {'milestone': 'H2', 'python': platform.python_version(), 'numpy': np.__version__,
              'scipy': scipy.__version__, 'platform': platform.platform(),
              'tree_count': len(trees), 'tree_prior': asdict(prior), 'tau2': sampler.tau2,
              'fixed_sigma2': sigma2, 'scale': asdict(sampler.scale),
              'burn_in': burn_in, 'draws_per_chain': draws, 'seeds': seeds,
              'row_normalization_error': row_error, 'detailed_balance_error': balance_error,
              'stationarity_error': stationary_error, 'spectral_radius_excluding_one': rho,
              'chains': chains, 'unknown_noise_signal_rmse': rmse,
              'unknown_noise_mean_sigma2': float(np.mean(learned.sigma2)),
              'elapsed_seconds': time.perf_counter() - started,
              'source_hashes': {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in source_paths},
              'scope': 'Exact reference uses m=1 and fixed variance; unknown-noise signal run is a smoke check'}
    (output / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')
    (output / 'encoder.json').write_text(json.dumps(encoder.to_dict(), indent=2) + '\n')
    structures = [{'index': i, 'tree': asdict(tree), 'posterior_probability': float(probability[i])}
                  for i, tree in enumerate(trees)]
    (output / 'structures.json').write_text(json.dumps(structures, indent=2) + '\n')
    np.savez_compressed(output / 'exact_reference.npz', y=y, probability=probability, kernel=matrix,
                        empirical_frequencies=frequencies, mean=mean, variance=variance)
    print(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('results/h2_validation'))
    parser.add_argument('--seeds', type=int, nargs='+', default=[1709, 1710, 1711, 1712])
    parser.add_argument('--draws', type=int, default=20000)
    parser.add_argument('--burn-in', type=int, default=2000)
    args = parser.parse_args()
    if args.draws < 1000 or args.burn_in < 0 or len(set(args.seeds)) != len(args.seeds):
        parser.error('Require >=1000 draws, nonnegative burn-in and distinct seeds')
    validate(args.output, args.seeds, args.draws, args.burn_in)


if __name__ == '__main__':
    main()
