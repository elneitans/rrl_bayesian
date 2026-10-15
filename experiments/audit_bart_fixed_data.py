"""Restart-free fixed-X/y BART sensitivity, four independent chains per job."""
import argparse
from dataclasses import asdict
import json
import io
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
    load_npz, manifest_encoder, read, sha, write_new, portable_metadata)
from experiments.pong_analysis_runtime import (Budget, WorkUnit, array_hash, atomic_bytes, atomic_json,
    checked_read, checked_write, extension_arguments, identity, mode_root, unit_lock)

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
    meta = portable_metadata(root/'fixed_fit')
    data = load_npz(root / 'fixed_fit/fixed_fit.npz', meta['npz_sha256'])
    transition_meta = portable_metadata(root/'fixed_transitions')
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
        'action_name': meta['actions'][action]['action'], 'calibration': saved,
        'kernel': meta['config'].get('kernel', 'grow_prune')}


def chain_identity(job, chain_seed, burn_in, draws):
    return {'data': job['data_hash'], 'values': array_hash(job['batch'].values),
        'observed': array_hash(job['batch'].observed), 'targets': array_hash(job['targets']),
        'catalog': job['batch'].catalog_hash, 'reference_values': array_hash(job['reference'].values),
        'reference_observed': array_hash(job['reference'].observed),
        'ids': job['transition_ids_sha256'], 'seed': chain_seed, 'burn_in': burn_in,
        'draws': draws, 'prior': asdict(job['prior']), 'noise': asdict(job['noise']),
        'k': job['k'], 'm': job['m'], 'kernel': job.get('kernel', 'grow_prune'),
        'initialization': job.get('initialization', 'stumps'), 'scale': asdict(job['scale']),
        'context': job.get('identity')}


def run_chain(job, output, *, chain_seed, burn_in, draws, deadline, checkpoint_every=100, extend_draws=0, extension_id=None):
    with unit_lock(output):
        return _run_chain(job, output, chain_seed=chain_seed, burn_in=burn_in, draws=draws,
                          deadline=deadline, checkpoint_every=checkpoint_every,
                          extend_draws=extend_draws, extension_id=extension_id)


def _run_chain(job, output, *, chain_seed, burn_in, draws, deadline, checkpoint_every, extend_draws, extension_id):
    unit = WorkUnit(output, chain_identity(job, chain_seed, burn_in, draws))
    saved = unit.load()
    length_path = output/'length.json'
    length = checked_read(length_path) if length_path.exists() else {'base_draws': draws, 'extensions': []}
    if type(extend_draws) is not int or extend_draws < 0 or (bool(extend_draws) != bool(extension_id)):
        raise ValueError('Draw extension requires positive integer and an explicit extension ID')
    if extend_draws:
        old = next((v for v in length['extensions'] if v['id'] == extension_id), None)
        if old is not None and old['draws'] != extend_draws:
            raise ValueError('Draw extension ID changed amount')
        if old is None:
            length['extensions'].append({'id': extension_id, 'draws': extend_draws, 'recorded_at_unix': time.time()})
    checked_write(length_path, length)
    total_draws = draws+sum(v['draws'] for v in length['extensions'])
    if saved and saved['status'] == 'completed' and saved['result']['retained'] >= total_draws:
        _publish_chain_view(unit, saved)
        return 'completed'
    regressor = BARTRegressor(m=job['m'], tree_prior=job['prior'],
        noise_prior=job['noise'], scale=job['scale'], k=job['k'], kernel=job.get('kernel', 'grow_prune'))
    rules = RuleCatalog(__import__('bayesian_rrtl.encoding', fromlist=['RelationalEncoder']).RelationalEncoder(job['batch'].catalog))
    from bayesian_rrtl.proposals import RevisionKernel
    kernel_class = {'grow_prune': GrowPruneKernel, 'revision': RevisionKernel}[job.get('kernel', 'grow_prune')]
    kernel = kernel_class(job['batch'], rules, job['prior'])
    y = job['scale'].normalize(job['targets'])
    if saved:
        state = unit.pickle(saved, 'progress')
        rng, cache, sigma2, traces = (state[k] for k in ('rng', 'cache', 'sigma2', 'traces'))
        elapsed = state['elapsed']
    else:
        rng = np.random.default_rng(chain_seed)
        trees = initial_trees(job, y, kernel, rng)
        cache = PredictionCache(trees, job['batch'])
        sigma2 = job['noise'].lambda_
        traces = {key: [] for key in ('sigma2', 'sse', 'leaves', 'depth', 'predictions', 'sampling_seconds',
                  *(f'{move}_{what}' for move in MOVES for what in ('selected', 'impossible', 'accepted', 'rejected')))}
        elapsed = 0.
    started = time.monotonic()

    def persist():
        state = {'rng': rng, 'cache': cache, 'sigma2': sigma2, 'traces': traces,
                 'elapsed': elapsed+time.monotonic()-started}
        unit.commit('partial', result={'sweeps': len(traces['sigma2']), 'execution': 'partial'},
                    artifacts={'progress': pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)})

    if not saved:
        persist()
    for sweep in range(len(traces['sigma2']), burn_in+total_draws):
        if time.monotonic() >= deadline:
            persist()
            return 'partial'
        sampling_started = time.perf_counter()
        sigma2, diagnostic, _ = regressor.sweep(cache, sigma2, y, kernel, rng)
        traces['sampling_seconds'].append(time.perf_counter()-sampling_started)
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
            traces[f'{move}_rejected'].append(sum(not u.accepted and not u.impossible for u in updates))
        if (sweep+1) % checkpoint_every == 0:
            persist()
    stream = io.BytesIO()
    np.savez_compressed(stream, **{key: np.asarray(value) for key, value in traces.items()})
    metadata = {'job_hash': digest(unit.expected), 'seed': chain_seed, 'sweeps': burn_in+total_draws,
                'burn_in': burn_in, 'retained': total_draws, 'seconds': elapsed+time.monotonic()-started,
                'trace_sha256': __import__('hashlib').sha256(stream.getvalue()).hexdigest()}
    end = {'rng': rng, 'cache': cache, 'sigma2': sigma2, 'traces': traces, 'elapsed': metadata['seconds']}
    saved = unit.commit('completed', result=metadata, artifacts={'trace': stream.getvalue(), 'progress': pickle.dumps(end)})
    _publish_chain_view(unit, saved)
    return 'completed'


def _publish_chain_view(unit, saved):
    # Compatibility views are regenerated exclusively from the validated commit.
    atomic_bytes(unit.folder/'trace.npz', unit.read_blob(saved['artifacts']['trace']))
    atomic_json(unit.folder/'metadata.json', saved['result'])


def initial_trees(job, y, kernel, rng):
    trees = [Tree(float(y.mean()/job['m'])) for _ in range(job['m'])]
    if job.get('initialization', 'stumps') == 'dispersed':
        from bayesian_rrtl.priors import leaf_variance
        sd = np.sqrt(leaf_variance(job['k'], job['m']))
        for i in range(job['m']):
            tree = Tree(float(rng.normal(0., sd)))
            grow, _ = kernel.choices(tree)
            if grow and i % 2 == 0:
                groups = grow[()]
                group = list(groups.values())[int(rng.integers(len(groups))) ]
                rule = group[int(rng.integers(len(group)))]
                tree = Tree(rule=rule, true_child=Tree(float(rng.normal(0., sd))),
                            false_child=Tree(float(rng.normal(0., sd))))
            tree.validate_support(job['batch'], kernel.rules, max_depth=job['prior'].max_depth,
                                  min_child=job['prior'].min_child)
            trees[i] = tree
    elif job.get('initialization', 'stumps') != 'stumps':
        raise ValueError('Unknown valid initialization')
    return trees


def observable_diagnostics(values, seconds, *, known_degenerate=False):
    values = np.asarray(values, dtype=float)
    record = chain_diagnostics(values)
    if record['status'] == 'constant':
        record['interpretation'] = 'known_support_degeneracy' if known_degenerate else 'nonconclusive_constant'
    elif record['status'] == 'stuck':
        record['interpretation'] = 'chains_stuck_at_different_constants'
    else:
        record['interpretation'] = 'operational_diagnostic'
    record['mean'] = float(values.mean())
    record['standard_deviation'] = float(values.std())
    record['median'] = float(np.median(values))
    record['posterior_interval_90'] = np.quantile(values, [.05, .95]).tolist()
    ess = record['ess_bulk']
    record['mcse_mean'] = float(values.std()/np.sqrt(ess)) if ess is not None and ess > 0 else None
    record['ess_bulk_per_second'] = ess/seconds if ess is not None and seconds > 0 else None
    record['ess_tail_per_second'] = record['ess_tail']/seconds if record['ess_tail'] is not None and seconds > 0 else None
    record['passes'] = (record['status'] == 'ok' and record['rhat'] <= 1.01 and
                        record['ess_bulk'] >= 400 and record['ess_tail'] >= 400)
    return record


def summarize(job, folder, chains, burn_in):
    arrays = []
    for i in range(chains):
        metadata = read(folder/f'chain_{i}/metadata.json')
        if sha(folder/f'chain_{i}/trace.npz') != metadata['trace_sha256']:
            raise ValueError('Corrupt MCMC trace view')
        arrays.append(np.load(folder/f'chain_{i}/trace.npz', allow_pickle=False))
    seconds = sum(read(folder/f'chain_{i}/metadata.json')['seconds'] for i in range(chains))
    retained = [len(a['sigma2'])-burn_in for a in arrays]
    if len(set(retained)) != 1:
        raise ValueError('Cannot summarize unequal/partial chain lengths')
    observables = {key: np.stack([array[key][burn_in:] for array in arrays]) for key in ('sigma2', 'sse', 'leaves', 'depth')}
    for i in range(len(job['reference'])):
        observables[f'prediction_{i}'] = np.stack([array['predictions'][burn_in:, i] for array in arrays])
    encoder = __import__('bayesian_rrtl.encoding', fromlist=['RelationalEncoder']).RelationalEncoder(job['batch'].catalog)
    rules = RuleCatalog(encoder)
    support_degenerate = not bool(job['prior'].eligible(rules, job['batch'], np.arange(len(job['batch'])), 0))
    diagnostics = {name: observable_diagnostics(values, seconds,
        known_degenerate=support_degenerate and name in ('leaves', 'depth')) for name, values in observables.items()}
    groups = {'predictive': {k: v for k, v in diagnostics.items() if k.startswith('prediction_')},
              'noise': {'sigma2_normalized': diagnostics['sigma2'],
                        'sigma2_original': observable_diagnostics(observables['sigma2']*job['scale'].width**2, seconds),
                        'sse_normalized': diagnostics['sse'],
                        'sse_original': observable_diagnostics(observables['sse']*job['scale'].width**2, seconds)},
              'structural': {k: diagnostics[k] for k in ('leaves', 'depth')}}
    moves = {move: {what: int(sum(array[f'{move}_{what}'].sum() for array in arrays))
                    for what in ('selected', 'impossible', 'accepted', 'rejected')} for move in MOVES}
    return {'status': 'completed', 'statistical_quality': 'satisfactory' if all(
        v['passes'] or v['interpretation']=='known_support_degeneracy' for v in diagnostics.values()) else 'insufficient',
        'seconds': seconds, 'sampling_seconds': float(sum(array['sampling_seconds'].sum() for array in arrays)),
        'diagnostics': diagnostics, 'diagnostic_groups': groups, 'retained_per_chain': retained[0],
        'all_observables_pass': all(v['passes'] for v in diagnostics.values()),
        'moves': moves, 'sigma2_median': float(np.median(observables['sigma2'])),
        'sigma2_original_median': float(np.median(observables['sigma2'])*job['scale'].width**2),
        'variance_conversion': 'original = width^2 * normalized',
        'data_hash': job['data_hash'], 'targets_sha256': array_hash(job['targets']),
        'transition_ids_sha256': job['transition_ids_sha256'], 'reference_rule': job['reference_rule'],
        'references_sha256': array_hash(job['reference'].values), 'catalog_hash': job['batch'].catalog_hash,
        'scale': asdict(job['scale']), 'tree_prior': asdict(job['prior']), 'noise_prior': asdict(job['noise']),
        'calibration': job.get('calibration'), 'calibration_method': job['calibration_method'],
        'variance_floor_used': job['variance_floor_used'], 'initialization': job.get('initialization', 'stumps'),
        'm': job['m'], 'kernel': job.get('kernel', 'grow_prune'),
        'criteria': {'rhat_max': 1.01, 'ess_bulk_min': 400, 'ess_tail_min': 400, 'absolute_guarantee': False},
        'limitation': 'Fixed-problem inference for this action only; not certification of online fits or six-action greedy decisions.'}


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
    parser.add_argument('--all-actions', action='store_true', help='Explicit six-action audit scope; no automatic grid')
    parser.add_argument('--condition', choices=tuple(CONDITIONS))
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--select-only', action='store_true')
    parser.add_argument('--initialization', choices=('stumps', 'dispersed'), default='stumps')
    parser.add_argument('--extend-draws', type=int, default=0)
    parser.add_argument('--review-base', help='Record explicit scientific review before enabling sensitivities')
    extension_arguments(parser)
    args = parser.parse_args()
    manifest = load_manifest(args.analysis, writable=True)
    specification = manifest['config']
    root = args.analysis / args.pipeline
    trans_meta = portable_metadata(root/'fixed_transitions')
    transitions = load_npz(root / 'fixed_transitions/fixed_transitions.npz', trans_meta['npz_sha256'])
    selection = select_actions(transitions, trans_meta['config']['actions'],
                               len(trans_meta['config']['actions']) if args.all_actions else specification['mcmc_initial_actions'])
    selection_path = root / ('selected_actions_all.json' if args.all_actions else 'selected_actions.json')
    if not selection_path.exists():
        write_new(selection_path, selection)
    elif read(selection_path) != selection:
        raise ValueError('Predeclared action selection changed')
    if args.select_only:
        print(json.dumps(selection, indent=2))
        return
    if args.action not in selection['actions'] or args.condition is None:
        parser.error('Select a predeclared action and condition')
    review_path = mode_root(args.analysis, args.pipeline, 'audit', args.smoke)/f'a{args.action}'/'base_review.json'
    if args.condition != 'base' and not review_path.exists():
        parser.error('Review the base with --condition base --review-base NOTE before enabling sensitivity')
    job = fixed_job(args.analysis, args.pipeline, args.action, args.condition)
    if args.condition != 'base':
        review = checked_read(review_path)
        if review['data_hash'] != job['data_hash'] or review['targets_sha256'] != array_hash(job['targets']):
            raise ValueError('Base review belongs to another fixed problem')
    job['initialization'] = args.initialization
    smoke = args.smoke
    burn_in = 10 if smoke else specification['mcmc_burn_in']
    draws = 20 if smoke else specification['mcmc_draws']
    folder = mode_root(args.analysis, args.pipeline, 'audit', smoke) / f'a{args.action}' / args.condition / args.initialization
    folder.mkdir(parents=True, exist_ok=True)
    report_path = folder / 'report.json'
    job['identity'] = identity(manifest, pipeline=args.pipeline, stage='audit', method='bart',
        dataset='pilot_fixed_fit', variant=args.condition, smoke=smoke,
        seeds=[specification['mcmc_seed']+args.action*100+i for i in range(specification['mcmc_chains'])],
        effective={'burn_in': burn_in, 'draws': draws},
        hashes={'data': job['data_hash'], 'targets': array_hash(job['targets']), 'catalog': job['batch'].catalog_hash},
        mcmc={'kernel': job.get('kernel', 'grow_prune'), 'm': job['m'], 'initialization': args.initialization,
              'prior': asdict(job['prior']), 'scale': asdict(job['scale'])})
    with unit_lock(folder):
        budget = Budget(folder, 120 if smoke else specification['mcmc_seconds_per_job'],
                        extend_seconds=args.extend_seconds, extension_id=args.extension_id if args.extend_seconds else None)
        try:
            _audit_report(args, job, folder, report_path, burn_in, draws, specification, budget)
        finally:
            budget.charge()


def _audit_report(args, job, folder, report_path, burn_in, draws, specification, budget):
    smoke = args.smoke
    deadline = time.monotonic()+budget.remaining
    states = []
    for chain in range(specification['mcmc_chains']):
        seed = specification['mcmc_seed'] + args.action * 100 + chain
        states.append(run_chain(job, folder / f'chain_{chain}', chain_seed=seed,
            burn_in=burn_in, draws=draws, deadline=deadline, checkpoint_every=10 if smoke else 100, extend_draws=args.extend_draws,
            extension_id=args.extension_id if args.extend_draws else None))
        if states[-1] == 'partial':
            break
    if len(states) == specification['mcmc_chains'] and set(states) == {'completed'}:
        report = summarize(job, folder, specification['mcmc_chains'], burn_in)
        plot_traces(folder, burn_in)
    else:
        report = {'status': 'partial', 'completed_chains': states.count('completed'),
                  'reason': 'time_budget', 'data_hash': job['data_hash']}
    if args.review_base:
        if args.condition != 'base' or report['status'] != 'completed':
            raise ValueError('Base review requires all completed base chains')
        review = {'note': args.review_base, 'report_sha256': digest(report), 'scientific_quality': report['statistical_quality'],
                  'data_hash': job['data_hash'], 'targets_sha256': array_hash(job['targets'])}
        checked_write(mode_root(args.analysis, args.pipeline, 'audit', smoke)/f'a{args.action}'/'base_review.json', review)
    report.update(pipeline=args.pipeline, action=args.action, action_name=job['action_name'],
        condition=args.condition, burn_in=burn_in, retained=report.get('retained_per_chain', None),
        requested_base_retained=draws,
        chains=specification['mcmc_chains'], smoke=smoke)
    atomic_json(report_path, report)
    comparison = compare_initializations(folder.parent)
    if comparison is not None:
        atomic_json(folder.parent/'initialization_comparison.json', comparison)
    print(json.dumps({'status': report['status'], 'action': args.action,
                      'condition': args.condition, 'smoke': smoke}))


def compare_initializations(condition_folder):
    """Compare verified chain commits, without equating agreement to validity."""
    folders = [condition_folder/name for name in ('stumps', 'dispersed')]
    if any(not (path/'report.json').exists() for path in folders):
        return None
    reports = [read(path/'report.json') for path in folders]
    if any(report['status'] != 'completed' for report in reports):
        return {'status': 'partial', 'interpretation': 'Both initialization jobs must complete.'}
    for folder, report in zip(folders, reports):
        for i in range(report['chains']):
            chain = folder/f'chain_{i}'
            with unit_lock(chain):
                unit = WorkUnit(chain, checked_read(chain/'identity.json'))
                saved = unit.load()
                if saved['status'] != 'completed' or saved['result']['retained'] != report['retained_per_chain']:
                    raise ValueError('Stale initialization report')
    for field in ('data_hash', 'targets_sha256', 'references_sha256', 'catalog_hash',
                  'scale', 'tree_prior', 'noise_prior', 'm', 'kernel', 'retained_per_chain'):
        if reports[0][field] != reports[1][field]:
            return {'status': 'incompatible', 'field': field,
                    'interpretation': 'Initialization comparison requires the same fixed problem and chain length.'}
    differences = {}
    for name, first in reports[0]['diagnostics'].items():
        second = reports[1]['diagnostics'][name]
        mcse = [first['mcse_mean'], second['mcse_mean']]
        se = float(np.hypot(*mcse)) if all(v is not None for v in mcse) else None
        differences[name] = {'mean_difference_dispersed_minus_stumps': second['mean']-first['mean'],
                             'combined_mcse': se}
    return {'status': 'checked', 'differences': differences,
            'statistical_quality': [r['statistical_quality'] for r in reports],
            'interpretation': 'Compare differences with Monte Carlo uncertainty and chain diagnostics; agreement alone does not certify convergence.'}


if __name__ == '__main__':
    main()
