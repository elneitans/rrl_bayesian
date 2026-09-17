"""Reproducible multi-chain BART validation on four fixed synthetic datasets."""

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

from bayesian_rrtl import RelationalEncoder
from bayesian_rrtl.diagnostics import posterior_diagnostics
from bayesian_rrtl.ensemble import BARTRegressor
from bayesian_rrtl.priors import NoisePrior, TreePrior

Fact = namedtuple('Fact', 'type value name obj1 obj2 obj3', defaults=[None, None])


def synthetic_problem(name):
    states = [[Fact('logical', str(x), 'more-x', 'player_t', 'ball_t'),
               Fact('logical', str(y), 'more-y', 'ball_t', 'ball_t-1')]
              for x, y in [(False, False), (False, True), (True, False), (True, True)]]
    signals = {'null': [0, 0, 0, 0], 'piecewise': [-0.3, -0.3, 0.3, 0.3],
               'interaction': [0.25, -0.25, -0.25, 0.25], 'rare': [0, 0, 0, 0.35]}
    counts = [40, 40, 40, 8] if name == 'rare' else [30, 30, 30, 30]
    encoder = RelationalEncoder.from_states(states)
    indices = np.repeat(np.arange(4), counts)
    truth = np.array(signals[name], dtype=float)
    data_seed = {'null': 300, 'piecewise': 301, 'interaction': 302, 'rare': 303}[name]
    y = truth[indices] + np.random.default_rng(data_seed).normal(0, 0.08, len(indices))
    return encoder, encoder.transform([states[i] for i in indices]), y, encoder.transform(states), truth, counts


def run(output, *, seeds, m, burn_in, draws, cases):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    model = BARTRegressor(m=m, tree_prior=TreePrior(alpha_tree=0.6, max_depth=2),
                          noise_prior=NoisePrior(nu=3, lambda_=0.01))
    reports = {}
    started = time.perf_counter()
    for name in cases:
        encoder, batch, y, reference, truth, counts = synthetic_problem(name)
        folder = output / name
        folder.mkdir()
        chains = []
        for seed in seeds:
            chain = model.fit(batch, y, np.random.default_rng(seed), burn_in=burn_in, draws=draws)
            chains.append(chain)
            print(json.dumps({'case': name, 'seed': seed, 'seconds': round(chain.elapsed_seconds, 2)}), flush=True)
        diagnostics = posterior_diagnostics(chains, reference)
        latent = np.array([chain.predict_latent_draws(reference) for chain in chains])
        mean = latent.mean(axis=(0, 1))
        rmse = float(np.sqrt(np.mean((mean - truth)**2)))
        assert np.isfinite(latent).all() and rmse < 0.10, f'{name}: signal recovery failed'
        assert np.max(np.abs(mean - truth)) < 0.15, f'{name}: a reference state was not recovered'
        # Record status honestly rather than interpreting acceptance as convergence.
        quality = all(d['status'] == 'ok' and d['rhat'] <= 1.01 and d['ess_bulk'] >= 400
                      for d in diagnostics.values())
        predictions = [v for k, v in diagnostics.items() if k.startswith('prediction_')]
        prediction_quality = all(d['status'] == 'ok' and d['rhat'] <= 1.01 and d['ess_bulk'] >= 400
                                 for d in predictions)
        report = {'counts_by_state': counts, 'signal': truth.tolist(), 'posterior_mean': mean.tolist(),
                  'rmse': rmse, 'max_state_error': float(np.max(np.abs(mean-truth))),
                  'all_observables_meet_rhat_1_01_bulk_ess_400': quality,
                  'predictions_meet_rhat_1_01_bulk_ess_400': prediction_quality,
                  'diagnostics': diagnostics, 'chains': [chain.diagnostics() for chain in chains],
                  'data_hash': chains[0].data_hash,
                  'interpretation': 'One fixed dataset, not a repeated-dataset coverage study'}
        reports[name] = report
        (folder / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        (folder / 'encoder.json').write_text(json.dumps(encoder.to_dict(), indent=2) + '\n')
        np.savez_compressed(folder / 'draws.npz', predictions=latent,
                            sigma2=np.array([chain.sigma2 for chain in chains]),
                            total_leaves=np.array([[s.total_leaves for s in c.trace[c.burn_in:]] for c in chains]),
                            max_depth=np.array([[s.max_depth for s in c.trace[c.burn_in:]] for c in chains]),
                            y=y, truth=truth, values=batch.values, observed=batch.observed)
        print(json.dumps({'case_complete': name, 'rmse': rmse, 'quality': quality}), flush=True)
    root = Path(__file__).resolve().parents[1]
    sources = [*sorted((root / 'bayesian_rrtl').glob('*.py')), Path(__file__).resolve()]
    report = {'milestone': 'H3', 'python': platform.python_version(), 'platform': platform.platform(),
              'numpy': np.__version__, 'scipy': scipy.__version__, 'm': m, 'seeds': seeds,
              'burn_in': burn_in, 'draws_per_chain': draws, 'tree_prior': asdict(model.tree_prior),
              'noise_prior': asdict(model.noise_prior), 'scale': asdict(model.scale), 'k': model.k,
              'tau2': model.tau2, 'cases': reports, 'elapsed_seconds': time.perf_counter()-started,
              'source_hashes': {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
              'scope': 'Offline synthetic regression; no Atari policy, coverage campaign or General BART'}
    (output / 'validation.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'output': str(output), 'elapsed_seconds': report['elapsed_seconds']}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('results/h3_validation'))
    parser.add_argument('--m', type=int, default=5)
    parser.add_argument('--seeds', type=int, nargs='+', default=[310, 311, 312, 313])
    parser.add_argument('--burn-in', type=int, default=1000)
    parser.add_argument('--draws', type=int, default=2000)
    parser.add_argument('--cases', nargs='+', choices=('null', 'piecewise', 'interaction', 'rare'),
                        default=['null', 'piecewise', 'interaction', 'rare'])
    args = parser.parse_args()
    if len(args.seeds) < 2 or len(set(args.seeds)) != len(args.seeds) or args.draws < 4 or args.burn_in < 0:
        parser.error('Need distinct seeds for >=2 chains, >=4 draws and nonnegative burn-in')
    run(args.output, seeds=args.seeds, m=args.m, burn_in=args.burn_in, draws=args.draws, cases=args.cases)


if __name__ == '__main__':
    main()
