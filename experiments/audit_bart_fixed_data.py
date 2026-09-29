"""Restart-free fixed-X/y BART sensitivity, four independent chains per job."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import pickle
import time

import numpy as np

from bayesian_rrtl.diagnostics import chain_diagnostics
from bayesian_rrtl.ensemble import BARTRegressor, PredictionCache
from bayesian_rrtl.priors import FixedScale, NoisePrior, TreePrior
from bayesian_rrtl.proposals import GrowPruneKernel
from bayesian_rrtl.rules import RuleCatalog
from bayesian_rrtl.tree import Tree
from experiments.pong_analysis_common import (digest, encoded, load_manifest,
    load_npz, manifest_encoder, read, save_npz, sha, write_new)

CONDITIONS = {'base': {}, 'alpha_08': {'alpha_tree': .8}, 'beta_1': {'beta_tree': 1.},
    'k_1': {'k': 1.}, 'k_3': {'k': 3.}, 'q_075': {'q': .75}, 'q_099': {'q': .99}}
MOVES = ('grow', 'prune', 'change', 'swap')


def select_actions(transition_data, actions, limit=3):
    counts = [int(np.sum(transition_data['actions'] == i)) for i in range(len(actions))]
    positives = [int(np.sum((transition_data['actions'] == i) &
                            (transition_data['raw_rewards'] > 0))) for i in range(len(actions))]
    high = max(range(len(actions)), key=lambda i: (counts[i], -i))
    low = min(range(len(actions)), key=lambda i: (counts[i], i))
    choice = [high]
    if low not in choice:
        choice.append(low)
    positive = max(range(len(actions)), key=lambda i: (positives[i], -i))
    if positives[positive] and positive not in choice:
        choice.append(positive)
    for i in range(len(actions)):
        if len(choice) >= limit:
            break
        if i not in choice:
            choice.append(i)
    return {'actions': choice[:limit], 'coverage': counts, 'positive_rewards': positives,
            'rule': 'max replay coverage, min replay coverage, then max positive raw rewards; index tie-break'}


def references(data, encoder):
    frequency = {}
    for values, observed in zip(data['state_values'], data['state_observed']):
        key = values.tobytes() + observed.tobytes()
        if key not in frequency:
            frequency[key] = [0, values.copy(), observed.copy()]
        frequency[key][0] += 1
    common = sorted(frequency, key=lambda k: (-frequency[k][0], k))[:2]
    rare = [k for k in sorted(frequency, key=lambda k: (frequency[k][0], k)) if k not in common][:2]
    keys = common + rare
    if len(keys) < 2:
        raise ValueError('Replay lacks distinct reference states')
    batch = encoded(np.stack([frequency[k][1] for k in keys]),
                    np.stack([frequency[k][2] for k in keys]), encoder)
    return batch, [{'frequency': frequency[k][0], 'group': 'frequent' if k in common else 'rare'}
                   for k in keys]


def fixed_job(analysis, pipeline, action, condition):
    root = Path(analysis) / pipeline
    meta = read(root / 'fixed_fit/metadata.json')
    data = load_npz(root / 'fixed_fit/fixed_fit.npz', meta['npz_sha256'])
    transition_meta = read(root / 'fixed_transitions/metadata.json')
    transitions = load_npz(root / 'fixed_transitions/fixed_transitions.npz', transition_meta['npz_sha256'])
    encoder = manifest_encoder(meta)
    prefix = f'a{action}_'
    batch = encoded(data[prefix+'values'], data[prefix+'observed'], encoder)
    targets = np.asarray(data[prefix+'targets'], dtype=float)
    if not len(targets):
        raise ValueError('Selected action has no last-fit data')
    reference, selected = references(transitions, encoder)
    params = CONDITIONS[condition]
    prior = TreePrior(**(meta['tree_prior'] | {k: v for k, v in params.items() if k in ('alpha_tree', 'beta_tree')}))
    scale = FixedScale(**meta['scale'])
    k = params.get('k', meta['config']['k'])
    saved = meta['actions'][action]['calibration']
    noise = NoisePrior(**saved['prior'])
    calibration_method = saved['method']
    if 'q' in params:
        from scipy.stats import chi2
        nu = meta['noise_prior_parameters']['nu']
        original_q = meta['noise_prior_parameters']['q']
        noise = NoisePrior(nu, noise.lambda_ * chi2.ppf(1-params['q'], nu)
                           / chi2.ppf(1-original_q, nu))
        calibration_method += '_same_pilot_variance_quantile_rescaled'
    return {'batch': batch, 'targets': targets, 'reference': reference, 'reference_rule': selected,
        'prior': prior, 'noise': noise, 'scale': scale, 'k': k, 'm': meta['config']['m'],
        'data_hash': meta['actions'][action]['data_hash'],
        'transition_ids_sha256': digest(data[prefix+'transition_ids'].tolist()),
        'fixed_fit_sha256': meta['npz_sha256'], 'calibration_method': calibration_method,
        'variance_floor_used': 'floor' in calibration_method,
        'action_name': meta['actions'][action]['action']}


def _progress(path, state):
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as handle:
        pickle.dump(state, handle)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def run_chain(job, output, *, chain_seed, burn_in, draws, deadline, checkpoint_every=100):
    """Persist exact cache/RNG continuation; each chain still starts cold."""
    output.mkdir(parents=True, exist_ok=True)
    finished = output / 'trace.npz'
    if finished.exists():
        return 'completed'
    progress = output / 'progress.pkl'
    regressor = BARTRegressor(m=job['m'], tree_prior=job['prior'],
        noise_prior=job['noise'], scale=job['scale'], k=job['k'], kernel='grow_prune')
    rules = RuleCatalog(__import__('bayesian_rrtl.encoding', fromlist=['RelationalEncoder']).RelationalEncoder(job['batch'].catalog))
    kernel = GrowPruneKernel(job['batch'], rules, job['prior'])
    y = job['scale'].normalize(job['targets'])
    job_hash = digest({'data': job['data_hash'], 'seed': chain_seed, 'burn_in': burn_in,
        'draws': draws, 'prior': asdict(job['prior']), 'noise': asdict(job['noise']),
        'k': job['k'], 'scale': asdict(job['scale'])})
    if progress.exists():
        with progress.open('rb') as handle:
            state = pickle.load(handle)  # trusted local job-owned progress only
        if state['job_hash'] != job_hash:
            raise ValueError('Incompatible chain progress')
        rng, cache, sigma2, traces = (state[k] for k in ('rng', 'cache', 'sigma2', 'traces'))
        start_sweep = len(traces['sigma2'])
        elapsed = state['elapsed']
    else:
        rng = np.random.default_rng(chain_seed)
        cache = PredictionCache([Tree(float(y.mean()/job['m'])) for _ in range(job['m'])], job['batch'])
        sigma2 = job['noise'].lambda_
        traces = {key: [] for key in ('sigma2', 'sse', 'leaves', 'depth', 'predictions',
                  *(f'{move}_{what}' for move in MOVES for what in ('selected', 'impossible', 'accepted')))}
        start_sweep = 0
        elapsed = 0.
    started = time.monotonic()
    for sweep in range(start_sweep, burn_in + draws):
        if time.monotonic() >= deadline:
            _progress(progress, {'job_hash': job_hash, 'rng': rng, 'cache': cache,
                'sigma2': sigma2, 'traces': traces, 'elapsed': elapsed + time.monotonic()-started})
            return 'partial'
        sigma2, diagnostic, _ = regressor.sweep(cache, sigma2, y, kernel, rng)
        traces['sigma2'].append(sigma2)
        traces['sse'].append(diagnostic.residual_sum_squares)
        traces['leaves'].append(diagnostic.total_leaves)
        traces['depth'].append(diagnostic.max_depth)
        prediction = job['scale'].restore(sum(tree.predict(job['reference']) for tree in cache.trees))
        traces['predictions'].append(prediction)
        for move in MOVES:
            updates = [u for u in diagnostic.updates if u.move == move]
            traces[f'{move}_selected'].append(len(updates))
            traces[f'{move}_impossible'].append(sum(u.impossible for u in updates))
            traces[f'{move}_accepted'].append(sum(u.accepted for u in updates))
        if (sweep + 1) % checkpoint_every == 0:
            _progress(progress, {'job_hash': job_hash, 'rng': rng, 'cache': cache,
                'sigma2': sigma2, 'traces': traces, 'elapsed': elapsed + time.monotonic()-started})
    seconds = elapsed + time.monotonic()-started
    save_npz(finished, **{key: np.asarray(value) for key, value in traces.items()})
    write_new(output / 'metadata.json', {'job_hash': job_hash, 'seed': chain_seed,
        'sweeps': burn_in + draws, 'burn_in': burn_in, 'retained': draws,
        'seconds': seconds, 'trace_sha256': sha(finished)})
    progress.unlink(missing_ok=True)
    return 'completed'


def summarize(job, folder, chains, burn_in):
    arrays = [np.load(folder / f'chain_{i}/trace.npz', allow_pickle=False) for i in range(chains)]
    seconds = sum(read(folder / f'chain_{i}/metadata.json')['seconds'] for i in range(chains))
    observables = {key: np.stack([array[key][burn_in:] for array in arrays])
        for key in ('sigma2', 'sse', 'leaves', 'depth')}
    for i in range(len(job['reference'])):
        observables[f'prediction_{i}'] = np.stack([array['predictions'][burn_in:, i] for array in arrays])
    diagnostics = {}
    for name, values in observables.items():
        record = chain_diagnostics(values)
        record['ess_bulk_per_second'] = (record['ess_bulk'] / seconds if record['ess_bulk'] is not None else None)
        record['ess_tail_per_second'] = (record['ess_tail'] / seconds if record['ess_tail'] is not None else None)
        record['passes'] = (record['status'] == 'ok' and record['rhat'] <= 1.01
                            and record['ess_bulk'] >= 400 and record['ess_tail'] >= 400)
        diagnostics[name] = record
    moves = {move: {what: int(sum(array[f'{move}_{what}'].sum() for array in arrays))
                    for what in ('selected', 'impossible', 'accepted')} for move in MOVES}
    return {'status': 'completed', 'seconds': seconds, 'diagnostics': diagnostics,
        'all_observables_pass': all(v['passes'] for v in diagnostics.values()),
        'moves': moves, 'sigma2_median': float(np.median(observables['sigma2'])),
        'sse_median': float(np.median(observables['sse'])),
        'reference_rule': job['reference_rule'], 'data_hash': job['data_hash'],
        'transition_ids_sha256': job['transition_ids_sha256'],
        'scale': asdict(job['scale']), 'tree_prior': asdict(job['prior']),
        'noise_prior': asdict(job['noise']), 'calibration_method': job['calibration_method'],
        'variance_floor_used': job['variance_floor_used']}


def plot_traces(folder, burn_in):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    arrays = sorted(folder.glob('chain_*/trace.npz'))
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    for path in arrays:
        data = np.load(path)
        name = path.parent.name
        for axis, variable in zip(axes.flat, ('sigma2', 'sse', 'leaves', 'depth')):
            axis.plot(data[variable], linewidth=.4, label=name)
            axis.axvline(burn_in, color='black', linewidth=.6)
            axis.set_title(variable)
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(folder / 'traces.png', dpi=110)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    for axis, variable in zip(axes.flat, ('sigma2', 'sse', 'leaves', 'depth')):
        for path in arrays:
            data = np.load(path)
            axis.hist(data[variable][burn_in:], bins=25, alpha=.35, label=path.parent.name)
        axis.set_title(variable)
    fig.tight_layout()
    fig.savefig(folder / 'histograms.png', dpi=110)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', type=Path, required=True)
    parser.add_argument('--pipeline', choices=('ocatari', 'visual'), required=True)
    parser.add_argument('--action', type=int)
    parser.add_argument('--condition', choices=tuple(CONDITIONS))
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--select-only', action='store_true')
    args = parser.parse_args()
    manifest = load_manifest(args.analysis)
    specification = manifest['config']
    root = args.analysis / args.pipeline
    trans_meta = read(root / 'fixed_transitions/metadata.json')
    transitions = load_npz(root / 'fixed_transitions/fixed_transitions.npz', trans_meta['npz_sha256'])
    selection = select_actions(transitions, trans_meta['config']['actions'],
                               specification['mcmc_initial_actions'])
    selection_path = root / 'selected_actions.json'
    if not selection_path.exists():
        write_new(selection_path, selection)
    elif read(selection_path) != selection:
        raise ValueError('Predeclared action selection changed')
    if args.select_only:
        print(json.dumps(selection, indent=2))
        return
    if args.action not in selection['actions'] or args.condition is None:
        parser.error('Select a predeclared action and condition')
    job = fixed_job(args.analysis, args.pipeline, args.action, args.condition)
    smoke = args.smoke
    burn_in = 10 if smoke else specification['mcmc_burn_in']
    draws = 20 if smoke else specification['mcmc_draws']
    folder = root / ('audit_smoke' if smoke else 'audit') / f'a{args.action}' / args.condition
    folder.mkdir(parents=True, exist_ok=True)
    report_path = folder / 'report.json'
    if report_path.exists():
        old = read(report_path)
        if old['status'] == 'completed':
            print(json.dumps(old))
            return
    deadline = time.monotonic() + (120 if smoke else specification['mcmc_seconds_per_job'])
    states = []
    for chain in range(specification['mcmc_chains']):
        seed = specification['mcmc_seed'] + args.action * 100 + chain
        states.append(run_chain(job, folder / f'chain_{chain}', chain_seed=seed,
            burn_in=burn_in, draws=draws, deadline=deadline, checkpoint_every=10 if smoke else 100))
        if states[-1] == 'partial':
            break
    if len(states) == specification['mcmc_chains'] and set(states) == {'completed'}:
        report = summarize(job, folder, specification['mcmc_chains'], burn_in)
        plot_traces(folder, burn_in)
    else:
        report = {'status': 'partial', 'completed_chains': states.count('completed'),
                  'reason': 'time_budget', 'data_hash': job['data_hash']}
    report.update(pipeline=args.pipeline, action=args.action, action_name=job['action_name'],
        condition=args.condition, burn_in=burn_in, retained=draws,
        chains=specification['mcmc_chains'], smoke=smoke)
    if report_path.exists():
        report_path.unlink()
    write_new(report_path, report)
    print(json.dumps({'status': report['status'], 'action': args.action,
                      'condition': args.condition, 'smoke': smoke}))


if __name__ == '__main__':
    main()
