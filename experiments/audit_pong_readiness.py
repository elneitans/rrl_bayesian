"""Bounded fixed-data kernel audit and pilot cost estimate; never evaluates test."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import pickle
import subprocess
import sys
import time

import numpy as np

from bayesian_rrtl.diagnostics import posterior_diagnostics
from bayesian_rrtl.ensemble import BARTRegressor
from bayesian_rrtl.replay import concatenate_states
from experiments.compare_pong import _finals, load_frozen, load_session, read, write
from bayesian_rrtl.comparison_protocol import ROOT, digest, file_hash, resolve_protocol


def baseline_signal_check():
    """Default pilot control parameters, fixed +/-1 terminal synthetic signal."""
    from bayesian_rrtl.baseline_adapter import BaselineLearnerConfig, RelationalBaselineAgent
    from bayesian_rrtl.games.pong import PONG_ACTIONS, PongRelationsV1
    from preprocessing import Fact
    config = BaselineLearnerConfig()
    agent = RelationalBaselineAgent(PONG_ACTIONS, PongRelationsV1().encoder(), config)
    first = None
    for i in range(2048):
        more = (i // 256) % 2
        state = [Fact('comparative', 'more' if more else 'less', 'x', 'player_t', 'ball_t')]
        agent.observe(state, 0, 1. if more else -1., None, True, False)
        if first is None and agent.n_splits:
            first = i + 1
    if not agent.n_splits:
        raise AssertionError('Default baseline cannot split the declared synthetic signal')
    return {'status': 'passed', 'config': asdict(config), 'transitions': 2048,
            'signal': 'action NOOP, x less/-1 and more/+1 alternating blocks of 256, terminal targets',
            'first_split_at': first, 'splits': agent.n_splits}


def reference_states(agent, count=4):
    """Two most frequent and two rarest unique replay states, tie by byte code."""
    occurrences = {}
    for transition in agent.replay.transitions:
        state = transition.state
        key = state.values.tobytes() + state.observed.tobytes()
        if key not in occurrences:
            occurrences[key] = [0, transition.transition_id, state]
        occurrences[key][0] += 1
    common = sorted(occurrences, key=lambda k: (-occurrences[k][0], k))[:count // 2]
    rare = [k for k in sorted(occurrences, key=lambda k: (occurrences[k][0], k)) if k not in common][:count - len(common)]
    keys = common + rare
    return concatenate_states([occurrences[k][2] for k in keys], agent.encoder), [
        {'frequency': occurrences[k][0], 'first_transition_id': occurrences[k][1],
         'group': 'frequent' if k in common else 'rare'} for k in keys]


def prepare_job(cell, checkpoint, output, *, existing_ocatari=None):
    session = load_session(cell, checkpoint)
    try:
        agent = session.agent
        selection = {'source': str(checkpoint), 'reuse': False}
        if cell['pipeline'] == 'ocatari' and existing_ocatari is not None and Path(existing_ocatari).exists():
            from bayesian_rrtl.checkpoint import load_checkpoint
            old, collector = load_checkpoint(existing_ocatari)
            ignored = {'seed', 'validation_seeds', 'test_seeds'}
            matching_config = all(getattr(old.config, key) == value for key, value in asdict(agent.config).items()
                                  if key not in ignored)
            if matching_config and old.encoder.to_dict() == agent.encoder.to_dict() and old.last_fit is not None:
                session.environment.validate_snapshot(collector['environment'])
                agent, checkpoint = old, Path(existing_ocatari)
                selection = {'source': str(checkpoint), 'reuse': True,
                             'compatibility': 'Same learner hyperparameters, encoder and validated environmental semantics; development seed differs.'}
            else:
                selection['existing_rejection'] = 'Existing fit has incompatible learner configuration or encoder'
        if agent.last_fit is None:
            raise ValueError('No fixed completed fit for audit')
        reference, rule = reference_states(agent)
        payload = {'config': agent.config, 'fit': agent.last_fit, 'calibrations': agent.calibrations,
                   'reference': reference, 'reference_selection': rule, 'checkpoint_sha256': file_hash(checkpoint),
                   'selection': selection}
        path = Path(output) / 'fixed_fit.pkl'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps(payload))
        write(Path(output) / 'selection.json', selection)
        return path
    finally:
        session.environment.close()


def diagnose_action(payload_path, action, kernel, output):
    payload = pickle.loads(Path(payload_path).read_bytes())  # trusted local audit artifacts
    config, fit = payload['config'], payload['fit']
    dataset = fit.datasets[action]
    if not len(dataset.batch):
        write(Path(output) / 'report.json', {'status': 'no_data', 'action': action, 'kernel': kernel})
        return
    prior = payload['calibrations'][action].prior
    regressor = BARTRegressor(kernel=kernel, m=config.m, tree_prior=config.tree_prior,
                             scale=config.scale, k=config.k, noise_prior=prior)
    posteriors = []
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    streams = np.random.SeedSequence(630001 + action).spawn(4)
    for chain in range(4):
        posterior = regressor.fit(dataset.batch, dataset.targets, np.random.default_rng(streams[chain]),
                                   burn_in=500, draws=1000)
        posteriors.append(posterior)
        # Completed chains survive budget interruption of a later chain.
        np.savez_compressed(output / f'chain_{chain}.npz',
            predictions=posterior.predict_latent_draws(payload['reference']), sigma2=posterior.sigma2,
            leaves=[t.total_leaves for t in posterior.trace[500:]], depth=[t.max_depth for t in posterior.trace[500:]])
        write(output / 'progress.json', {'status': 'partial', 'completed_chains': chain + 1})
    diagnostics = posterior_diagnostics(posteriors, payload['reference'])
    seconds = time.perf_counter() - started
    for item in diagnostics.values():
        item['ess_bulk_per_second'] = item['ess_bulk'] / seconds if item['ess_bulk'] is not None else None
        item['ess_tail_per_second'] = item['ess_tail'] / seconds if item['ess_tail'] is not None else None
        item['passes_thresholds'] = (item['status'] == 'ok' and item['rhat'] <= 1.01
                                    and item['ess_bulk'] >= 400 and item['ess_tail'] >= 400)
    np.savez_compressed(output / 'fixed_data.npz', targets=dataset.targets, values=dataset.batch.values,
                       observed=dataset.batch.observed, transition_ids=dataset.transition_ids,
                       reference_values=payload['reference'].values, reference_observed=payload['reference'].observed)
    write(output / 'report.json', {'status': 'checked', 'kernel': kernel, 'action': config.actions[action],
        'chains': 4, 'burn_in': 500, 'draws': 1000, 'seconds': seconds, 'seed': 630001 + action,
        'fit_id': fit.fit_id, 'dataset_hash': dataset.data_hash, 'noise_prior': asdict(prior),
        'scale': asdict(config.scale), 'reference_selection': payload['reference_selection'],
        'checkpoint_sha256': payload['checkpoint_sha256'], 'diagnostics': diagnostics,
        'all_observables_pass': all(v['passes_thresholds'] for v in diagnostics.values()),
        'constant_observables': [k for k, v in diagnostics.items() if v['status'] == 'constant'],
        'chain_diagnostics': [p.diagnostics() for p in posteriors]})


def estimate_cost(manifest, summaries, pilot):
    """Explicit extrapolation, not a hardware-independent promise."""
    p = resolve_protocol(pilot)['protocol']
    smoke = manifest['protocol']
    lines = []
    for cell, summary, evaluations in summaries:
        # Scale BART training by fit count, trees, sweeps and batch size. This is
        # a stated approximation; deeper trees and replay can alter runtime.
        factor = p['interactions'] / smoke['interactions']
        if cell['learner'] == 'bart':
            old, new = smoke['learners']['bart'], p['learners']['bart']
            factor *= old['fit_interval'] / new['fit_interval'] * new['m'] / old['m']
            factor *= (new['burn_in'] + new['draws']) / (old['burn_in'] + old['draws'])
            factor *= new['batch_size_per_action'] / old['batch_size_per_action']
        measured_steps = sum(row['steps'] for result in evaluations for row in result['report']['episodes'])
        measured_episodes = sum(len(result['report']['episodes']) for result in evaluations)
        per_step = sum(result['seconds'] for result in evaluations) / measured_steps
        mean_steps = measured_steps / measured_episodes
        inference_factor = 1.
        if cell['learner'] == 'bart':
            inference_factor = (p['learners']['bart']['m'] * p['learners']['bart']['draws'] /
                                (smoke['learners']['bart']['m'] * smoke['learners']['bart']['draws']))
        episodes = len(p['curve_checkpoints']) * p['evaluation_episodes'] * len(p['training_seeds'])
        train = summary['timing']['training_seconds'] * factor * len(p['training_seeds'])
        lines.append({'pipeline': cell['pipeline'], 'learner': cell['learner'],
            'training_seconds_estimate': train, 'validation_episodes': episodes,
            'measured_evaluation_seconds_per_decision': per_step,
            'central_validation_seconds': per_step * mean_steps * episodes * inference_factor,
            'horizon_validation_seconds': per_step * p['evaluation_horizon'] * episodes * inference_factor,
            'save_seconds_estimate': summary['timing']['save_seconds'] * factor * len(p['training_seeds']),
            'training_scaling_factor': factor, 'inference_scaling_factor': inference_factor})
    return {'cells': lines, 'training_runs': 12, 'validation_episodes': 600,
        'central_seconds': sum(v['training_seconds_estimate'] + v['central_validation_seconds'] + v['save_seconds_estimate'] for v in lines),
        'horizon_seconds': sum(v['training_seconds_estimate'] + v['horizon_validation_seconds'] + v['save_seconds_estimate'] for v in lines),
        'assumptions': ['Sequential on measured host; smoke episode lengths for central estimate are horizon-censored.',
            'Horizon estimate uses 27000 decisions each; not a strict runtime guarantee.',
            'BART cost scales with fits*trees*sweeps*batch; inference with trees*draws. Replay/tree geometry may increase cost.',
            '600 validation episodes included; no reserved test. No pilot has been run.']}


def audit(campaign, output, pilot, *, budget_seconds=1200):
    if not 0 < budget_seconds <= 1200:
        raise ValueError('Offline budget must be in (0,1200] seconds')
    campaign, output = Path(campaign), Path(output)
    manifest = load_frozen(campaign)
    finals = list(_finals(campaign, manifest))
    output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'partial', 'budget_seconds': budget_seconds, 'jobs': {}, 'baseline': {},
              'protocol_hash': digest(manifest), 'audit_source_sha256': file_hash(__file__), 'interpretation': [
                  'Fixed supervised data/prior; these diagnostics do not certify online fits or determine a winner.',
                  'Constant observables are reported separately and do not establish convergence.',
                  'Failed mixing: next diagnose chain length, rare-state geometry and scale/calibration; no automatic retuning.']}
    summaries, jobs = [], []
    started = time.monotonic()
    report['baseline_synthetic'] = baseline_signal_check()
    for cell, folder, checkpoint, summary in finals:
        evaluations = [read(folder / f'validation_{point}.json') for point in manifest['protocol']['curve_checkpoints']]
        summaries.append((cell, summary, evaluations))
        if cell['learner'] == 'corrected_common':
            report['baseline'][cell['pipeline']] = summary['metrics']['complexity']
        elif cell['seed'] == manifest['protocol']['training_seeds'][0]:
            fixed = prepare_job(cell, checkpoint, output / cell['pipeline'],
                existing_ocatari=ROOT / 'results/ocatari_pong_validation_step45/smoke/final_checkpoint.rrtl')
            jobs.append((cell['pipeline'], fixed))
    report['cost_estimate'] = estimate_cost(manifest, summaries, read(pilot))
    report['checkpoint_selection'] = {pipeline: read(output / pipeline / 'selection.json') for pipeline, _ in jobs}
    for pipeline, fixed in jobs:
        for action in range(6):
            for kernel in ('grow_prune', 'revision'):
                name = f'{pipeline}/{action}/{kernel}'
                folder = output / name
                remaining = budget_seconds - (time.monotonic() - started)
                if remaining <= 0:
                    report['jobs'][name] = {'status': 'not_run', 'reason': 'offline_budget_exhausted'}
                    continue
                folder.mkdir(parents=True, exist_ok=True)
                command = [sys.executable, '-m', 'experiments.audit_pong_readiness', '--worker', str(fixed),
                           '--action', str(action), '--kernel', kernel, '--output', str(folder)]
                with (folder / 'worker.log').open('w') as log:
                    try:
                        result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=remaining)
                        status = read(folder / 'report.json') if result.returncode == 0 else {'status': 'failed', 'reason': 'see worker.log'}
                    except subprocess.TimeoutExpired:
                        status = {'status': 'partial', 'reason': 'offline_budget_exhausted',
                                  'completed_chains': read(folder / 'progress.json')['completed_chains'] if (folder / 'progress.json').exists() else 0}
                report['jobs'][name] = status
                report['seconds'] = time.monotonic() - started
                write(output / 'report.json', report)
                print(json.dumps({'job': name, 'status': status['status'], 'seconds': report['seconds']}), flush=True)
    report['seconds'] = time.monotonic() - started
    report['status'] = 'completed' if all(v['status'] in ('checked', 'no_data') for v in report['jobs'].values()) else 'partial'
    report['mixing_passed_jobs'] = sum(v.get('all_observables_pass', False) for v in report['jobs'].values())
    report['pilot_executed'] = report['reserved_test_executed'] = False
    write(output / 'report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pilot', type=Path, default=ROOT / 'configs/comparison_pong_pilot.json')
    parser.add_argument('--budget-seconds', type=float, default=1200)
    parser.add_argument('--worker', type=Path)
    parser.add_argument('--action', type=int)
    parser.add_argument('--kernel', choices=('grow_prune', 'revision'))
    args = parser.parse_args()
    if args.worker:
        diagnose_action(args.worker, args.action, args.kernel, args.output)
    else:
        if args.campaign is None:
            parser.error('--campaign required')
        audit(args.campaign, args.output, args.pilot, budget_seconds=args.budget_seconds)


if __name__ == '__main__':
    main()
