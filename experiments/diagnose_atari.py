"""Independent chains on the exact fixed final-fit Atari datasets (no new replay)."""

import argparse
import json
from pathlib import Path
import time

import numpy as np

from bayesian_rrtl.checkpoint import load_checkpoint
from bayesian_rrtl.diagnostics import posterior_diagnostics
from bayesian_rrtl.ensemble import BARTRegressor


def diagnose(checkpoint, output, *, chains=4, burn_in=100, draws=100, seed=600001):
    if chains < 2 or draws < 4:
        raise ValueError('Diagnostics need >=2 chains and >=4 retained draws')
    agent, _ = load_checkpoint(checkpoint)
    if agent.last_fit is None:
        raise ValueError('Checkpoint has no completed fit to diagnose')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    config = agent.config
    streams = np.random.SeedSequence(seed).spawn(len(config.actions) * chains)
    report = {'config_hash': config.config_hash, 'fit_id': agent.last_fit.fit_id,
              'target_model_id': agent.last_fit.target_model_id,
              'chains': chains, 'burn_in': burn_in, 'draws': draws, 'seed': seed, 'actions': {}}
    started = time.perf_counter()
    for dataset in agent.last_fit.datasets:
        action = dataset.action
        name = config.actions[action]
        if not len(dataset.batch):
            report['actions'][name] = {'status': 'no_data'}
            continue
        model = BARTRegressor(kernel=config.kernel, m=config.m, tree_prior=config.tree_prior, scale=config.scale,
                             k=config.k, noise_prior=agent.calibrations[action].prior)
        reference = dataset.batch.take(np.unique(dataset.batch.values, axis=0, return_index=True)[1][:8])
        posteriors = []
        for chain in range(chains):
            posterior = model.fit(dataset.batch, dataset.targets,
                np.random.default_rng(streams[action * chains + chain]), burn_in=burn_in, draws=draws)
            posteriors.append(posterior)
            print(json.dumps({'action': name, 'chain': chain, 'seconds': posterior.elapsed_seconds}), flush=True)
        diagnostics = posterior_diagnostics(posteriors, reference)
        quality = all(d['status'] == 'ok' and d['rhat'] <= 1.01 and
                      d['ess_bulk'] >= 400 and d['ess_tail'] >= 400 for d in diagnostics.values())
        report['actions'][name] = {'status': 'checked', 'samples': len(dataset.batch),
            'dataset_hash': dataset.data_hash, 'posterior_data_hash': posteriors[0].data_hash,
            'rhat_1_01_ess_400_all_observables': quality, 'diagnostics': diagnostics,
            'chains': [p.diagnostics() for p in posteriors]}
        np.savez_compressed(output / f'{name}_draws.npz',
            predictions=np.stack([p.predict_latent_draws(reference) for p in posteriors]),
            sigma2=np.array([p.sigma2 for p in posteriors]),
            leaves=np.array([[s.total_leaves for s in p.trace[p.burn_in:]] for p in posteriors]),
            depth=np.array([[s.max_depth for s in p.trace[p.burn_in:]] for p in posteriors]),
            targets=dataset.targets, values=dataset.batch.values, observed=dataset.batch.observed,
            transition_ids=dataset.transition_ids, reference_values=reference.values)
        (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    report['seconds'] = time.perf_counter() - started
    report['interpretation'] = ('Supervised diagnostics on a frozen RL dataset; short chains are approximate. '
                                'Constant observables do not establish convergence. No new data or test calibration.')
    (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--chains', type=int, default=4)
    parser.add_argument('--burn-in', type=int, default=100)
    parser.add_argument('--draws', type=int, default=100)
    parser.add_argument('--seed', type=int, default=600001)
    args = parser.parse_args()
    diagnose(args.checkpoint, args.output, chains=args.chains, burn_in=args.burn_in,
             draws=args.draws, seed=args.seed)
