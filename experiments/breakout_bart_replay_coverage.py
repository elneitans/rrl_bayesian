"""Paired frozen BART refits and the matching online 1,024-decision control."""
import argparse
from collections import Counter, deque
from dataclasses import replace
import hashlib
import itertools
import json
from pathlib import Path
import pickle
from statistics import mean

import numpy as np

from bayesian_rrtl.agent import BayesianQModel
from bayesian_rrtl.ensemble import BARTRegressor
from bayesian_rrtl.online.multigame_protocol import raw_protocol
from bayesian_rrtl.online.protocol import authorize, freeze, load, select_cell
from bayesian_rrtl.online.storage import binding, checkpoint_metadata, stream_rows, verified_bytes
from bayesian_rrtl.replay import Transition, concatenate_states
from bayesian_rrtl.revision_diagnostics import behavior, verify_start
from bayesian_rrtl.rrtl_followup import PHASES, make_phase_bank
from bayesian_rrtl.targets import frozen_q_targets
from experiments.breakout_bart_ablation import ServiceController, service_summary
from experiments.breakout_bart_stability import protocols, run_cell
from experiments.compare_pong_progressive import StopRequest, default_factory
from experiments.diagnose_rrtl_improvements import verify_bank
from experiments.diagnose_rrtl_revision import blob_read, blob_write
from experiments.pong_analysis_runtime import checked_read, checked_write, digest, unit_lock

CONTROL = 'control_1024'
CONDITIONS = ('original', 'phase_coverage')
BUDGETS = {'standard': (50, 50), 'extended': (200, 300)}
POINTS = (300000, 400000, 500000)


def load_config(path):
    config = json.loads(path.read_text())
    if (config['points'] != list(POINTS) or config['chains'] < 3 or config['starts'] % 3
            or config['starts'] < 6 or config['horizon'] < 32
            or set(config['quota_per_action']) != set(PHASES)
            or any(n < 1 for n in config['quota_per_action'].values())
            or sum(config['quota_per_action'].values()) > 128):
        raise ValueError('Invalid coverage experiment design')
    return config


def checkpoint_agent(historical, manifest, cell, point):
    folder = historical / cell['id']
    path, = (folder / 'checkpoints').glob(f'{point}-*.pkl')
    ref = {'path': str(path.relative_to(folder)), 'sha256': path.stem.split('-', 1)[1]}
    meta = checkpoint_metadata(folder, ref)
    payload = pickle.loads(verified_bytes(folder, ref))
    if (meta['binding'] != binding(manifest, cell) or any(payload[k] != meta[k] for k in
            ('binding', 'counter', 'log_head')) or payload['counter'] != point):
        raise ValueError('Retained checkpoint does not match its immutable metadata')
    agent = payload['session']['collector']['agent']
    if (agent.interactions != point or agent.config.seed != cell['seed']
            or agent.model.model_id != agent.last_fit.fit_id
            or agent.target_model.model_id != agent.last_fit.target_model_id
            or agent.estimator_config.get('warm_start') or len(agent.last_fit.datasets) != 4):
        raise ValueError('Expected a fitted cold-start BART checkpoint')
    return agent, ref, payload['log_head']


def sample_sha(ids, batch, targets):
    raw = (batch.catalog_hash.encode() + np.asarray(ids, dtype='<i8').tobytes()
           + batch.values.tobytes() + batch.observed.tobytes() + targets.astype('<f8').tobytes())
    return hashlib.sha256(raw).hexdigest()


def has_ball(state):
    return any(f.name == 'present' and f.obj1 == 'ball_t' and str(f.value) == 'True' for f in state)


def replay_at_fits(folder, head, environment, encoder, fit_points, capacity, max_episode_steps):
    """Replay logged actions in ALE so post-loss states use actual ALE life losses."""
    frames = deque(maxlen=capacity)
    begin, last = min(fit_points) - capacity - max_episode_steps, max(fit_points)
    active_episode = None
    for row in stream_rows(folder, head):
        step = row['interaction']
        if step < begin:
            continue
        if step > last:
            break
        if row['step_id'] == 0:
            state = environment.reset(seed=row['episode_seed'])
            batch = encoder.transform([state])
            active_episode = row['episode_id']
            pending = 'initial_serve'
        if active_episode != row['episode_id']:
            if active_episode is None:
                continue
            raise ValueError('Logged episode lacks its reset transition')
        if has_ball(state):
            pending = None
            phase = 'in_play'
        else:
            phase = pending or 'other'
        following, reward, bruto, terminal, truncated = environment.step(row['action'])
        if (reward, bruto, terminal, truncated) != (row['reward'], row['raw_reward'], row['terminated'], row['truncated']):
            raise ValueError(f'ALE replay differs at interaction {step}')
        next_batch = None if terminal else encoder.transform([following])
        transition = Transition(row['transition_id'], batch, row['action'], reward, bruto,
                                next_batch, terminal, truncated, row['episode_id'], row['step_id'])
        frames.append((transition, phase))
        state, batch = following, next_batch
        if environment.last_auxiliary['life_lost']:
            pending = 'after_life_loss'
        if step in fit_points:
            if len(frames) != capacity:
                raise ValueError('Could not reconstruct complete fit-time replay')
            yield step, list(frames)


def cover_batch(original_ids, available, quota, rng):
    """Make the minimum substitutions needed for all feasible phase floors."""
    if len(set(original_ids)) != len(original_ids) or not set(original_ids) <= set(available):
        raise ValueError('Original batch IDs are missing or duplicated')
    selected = set(original_ids)
    before = Counter(available[i] for i in selected)
    available_counts = Counter(available.values())
    floor = {phase: min(n, available_counts[phase]) for phase, n in quota.items()}
    drop = []
    for phase in sorted(set(available.values())):
        excess = max(0, before[phase] - floor.get(phase, 0))
        choices = sorted(i for i in selected if available[i] == phase)
        drop.extend(int(i) for i in rng.choice(choices, excess, replace=False))
    added = []
    for phase, minimum in floor.items():
        need = max(0, minimum - before[phase])
        choices = sorted(i for i, label in available.items() if label == phase and i not in selected)
        if need:
            added.extend(int(i) for i in rng.choice(choices, need, replace=False))
    selected.difference_update(int(i) for i in rng.choice(drop, len(added), replace=False))
    selected.update(added)
    after = Counter(available[i] for i in selected)
    if len(selected) != len(original_ids) or any(after[p] < n for p, n in floor.items()):
        raise ValueError('Coverage batch changed its size or missed an attainable floor')
    return sorted(selected), {'available': dict(available_counts), 'original': dict(before),
                              'phase_coverage': dict(after), 'requested': quota, 'attainable': floor,
                              'substitutions': len(added)}


def batches(agent, frames, config):
    replay = {record.transition_id: (record, phase) for record, phase in frames}
    result = {name: [] for name in CONDITIONS}
    provenance = {}
    for stored in agent.last_fit.datasets:
        action = stored.action
        available = {i: phase for i, (t, phase) in replay.items() if t.action == action}
        if len(stored.transition_ids) != agent.config.batch_size_per_action:
            raise ValueError('Expected full and equally sized per-action batches')
        rng = np.random.default_rng([config['lot_seed'], agent.config.seed, agent.last_fit.interaction, action])
        covered, counts = cover_batch(stored.transition_ids, available, config['quota_per_action'], rng)
        provenance[agent.config.actions[action]] = counts
        for name, ids in zip(CONDITIONS, (stored.transition_ids, covered)):
            transitions = [replay[i][0] for i in ids]
            x = concatenate_states([t.state for t in transitions], agent.encoder)
            y = frozen_q_targets(transitions, agent.target_model, agent.encoder, gamma=agent.config.gamma,
                                 bootstrap_on_truncation=agent.config.bootstrap_on_truncation)
            sha = sample_sha(ids, x, y)
            if name == 'original' and sha != stored.data_hash:
                if (not np.array_equal(x.values, stored.batch.values)
                        or not np.array_equal(x.observed, stored.batch.observed)
                        or not np.allclose(y, stored.targets, atol=1e-12, rtol=0)
                        or sample_sha(ids, stored.batch, stored.targets) != stored.data_hash):
                    raise ValueError('Original fit dataset cannot be reproduced')
                y, sha = stored.targets, stored.data_hash
            result[name].append((x, y))
            counts[name + '_hash'] = sha
    return result, provenance


def make_panel(environment, bank, encoder, seed):
    """Collect a fixed panel from the held-out starts and a fixed behavior policy."""
    states = {phase: [] for phase in PHASES}
    for start in bank['starts']:
        state = verify_start(environment, start)
        pending = start['phase'] if start['phase'] != 'in_play' else None
        rng = np.random.default_rng([seed, start['id']])
        states[start['phase']].append(encoder.transform([state]))
        for step in range(96):
            action = behavior(state, rng)
            state, _, _, terminal, truncated = environment.step(action)
            if terminal or truncated:
                break
            if environment.last_auxiliary['life_lost']:
                pending = 'after_life_loss'
            if has_ball(state):
                pending = None
                phase = 'in_play'
            else:
                phase = pending
            if phase is not None and step % 3 == 0:
                states[phase].append(encoder.transform([state]))
    return {phase: concatenate_states(rows, encoder) for phase, rows in states.items()}


def launch_experiment(output, historical, stability, stability_config, config):
    history = checked_read(historical / 'campaign.json')
    design = json.loads(stability_config.read_text())
    control = protocols(design, history)['slow_target']
    control['name'] = config['name'] + '/' + CONTROL
    control['learners']['bart'].pop('target_update_interval')
    if control['learners']['bart'] != raw_protocol(history)['learners']['bart']:
        raise ValueError('Control estimator differs from the prior cadence reference')
    for variant in ('bounded_structure', 'slow_target'):
        comparison = raw_protocol(checked_read(stability / 'pilot' / variant / 'campaign.json'))
        for key in ('games', 'learning', 'exploration', 'cells'):
            if comparison[key] != control[key]:
                raise ValueError(f'Control does not match {variant} on {key}')
    freeze(control, output / CONTROL)
    authorize(output / CONTROL, 1000000, 'Paired BART cadence control requested by user.', game='Breakout')
    for cell in history['cells']:
        for point in config['points']:
            checkpoint_agent(historical, history, cell, point)
    cell = next(c for c in history['cells'] if c['seed'] == 733102)
    diagnostic_agent, _, head = checkpoint_agent(historical, history, cell, config['points'][0])
    environment = default_factory(cell)()
    try:
        bank = make_phase_bank(environment, count=config['starts'], seed=config['bank_seed'])
        verify_bank(environment, bank)
        panel = make_panel(environment, bank, environment.encoder(), config['bank_seed'])
        fit = diagnostic_agent.last_fit.interaction
        recovered = replay_at_fits(historical / cell['id'], head, environment, diagnostic_agent.encoder,
                                   {fit}, diagnostic_agent.config.replay_capacity,
                                   cell['environment']['horizon'])
        _, frames = next(recovered)
        _, smoke_batches = batches(diagnostic_agent, frames, config)
    finally:
        environment.close()
    reference = blob_write(output, 'coverage_bank', {'bank': bank, 'panel': panel})
    checked_write(output / 'preflight.json', {
        'config': config, 'source_campaign': digest(history), 'bank': reference,
        'training_seeds': design['training_seeds'], 'checkpoints': design['checkpoints'],
        'starts': dict(Counter(s['phase'] for s in bank['starts'])),
        'panel': {phase: len(batch) for phase, batch in panel.items()},
        'smoke_checkpoint': {'seed': cell['seed'], 'fit': fit, 'batches': smoke_batches},
        'reserved_test_episodes': 0,
    })


def policies(agent, action_batches, config, point, chain):
    windows = {name: [] for name in BUDGETS}
    for action, (batch, targets) in enumerate(action_batches):
        previous = agent.model.posteriors[action]
        regressor = BARTRegressor(kernel=agent.estimator_config['kernel'], m=previous.m,
                                  tree_prior=previous.tree_prior, scale=previous.scale,
                                  k=previous.k, noise_prior=previous.noise_prior)
        rng = np.random.default_rng([config['mcmc_seed'], agent.config.seed, point, chain, action])
        short_burn, short_draws = BUDGETS['standard']
        long_burn, long_draws = BUDGETS['extended']
        posterior = regressor.fit(batch, targets, rng, burn_in=short_burn,
                                  draws=long_burn + long_draws - short_burn)
        windows['standard'].append(replace(posterior, draws=posterior.draws[:short_draws],
                                           trace=posterior.trace[:short_burn + short_draws]))
        windows['extended'].append(replace(posterior, draws=posterior.draws[long_burn - short_burn:],
                                           burn_in=long_burn))
    return {name: BayesianQModel(agent.config.actions, tuple(posts), agent.model.catalog_hash)
            for name, posts in windows.items()}


def rollout(agent, model, environment, start, horizon, eval_seed, stop):
    controller = ServiceController(agent)
    controller.model = model
    controller.pending = {'initial_serve': ('reset', environment.steps),
                          'after_life_loss': ('life_lost', environment.steps)}.get(start['phase'])
    state = verify_start(environment, start)
    if controller.pending:
        controller.pending = (controller.pending[0], environment.steps)
    rng = np.random.default_rng([eval_seed, start['id']])
    score = 0.
    for i in range(horizon):
        if stop.requested:
            raise InterruptedError('Diagnostic interrupted')
        action = controller.action(state, environment, rng)
        following, reward, bruto, terminal, truncated = environment.step(action)
        controller.transition(state, action, following, reward, bruto, terminal, truncated, environment)
        score += bruto
        state = following
        if terminal or truncated:
            break
    if controller.pending:
        kind, step = controller.pending
        controller.records.append({'kind': kind, 'start': step, 'decisions': environment.steps - step, 'fired': False})
    return {'phase': start['phase'], 'start': start['id'], 'return': score, 'steps': i + 1,
            'terminated': terminal, 'truncated': not terminal and (truncated or i + 1 == horizon),
            'service': controller.records}


def diagnostics(agent, model, environment, bank, panel, config, stop):
    fire = model.actions.index('FIRE')
    other = [i for i in range(len(model.actions)) if i != fire]
    summary = {}
    for phase, batch in panel.items():
        q = model.predict(batch)
        margins = q[:, fire] - q[:, other].max(axis=1)
        summary[phase] = {'fire_margin': margins.tolist(), 'greedy': q.argmax(axis=1).tolist(),
                          'top_margin': (np.sort(q, axis=1)[:, -1] - np.sort(q, axis=1)[:, -2]).tolist()}
    episodes = [rollout(agent, model, environment, s, config['horizon'], config['eval_seed'], stop)
                for s in bank['starts']]
    return {'panel': summary, 'episodes': episodes}


def coverage(output, historical, config, index):
    preflight = checked_read(output / 'preflight.json')
    history = checked_read(historical / 'campaign.json')
    if preflight['config'] != config or preflight['source_campaign'] != digest(history):
        raise ValueError('Coverage inputs changed after preflight')
    cell = next(c for c in history['cells'] if c['seed'] == preflight['training_seeds'][index])
    seed = cell['seed']
    destination = output / 'coverage' / str(seed)
    bank_and_panel = blob_read(output, preflight['bank'])
    bank, panel = bank_and_panel['bank'], bank_and_panel['panel']
    checkpoints = [checkpoint_agent(historical, history, cell, point) for point in config['points']]
    at_fit = {agent.last_fit.interaction: (point, agent, ref) for point, (agent, ref, _) in
              zip(config['points'], checkpoints)}
    _, _, head = checkpoints[-1]
    replay_env = default_factory(cell)()
    eval_env = default_factory(cell)()
    try:
        with unit_lock(destination), StopRequest() as stop:
            first_agent = checkpoints[0][0]
            capacity = first_agent.config.replay_capacity
            replays = replay_at_fits(historical / cell['id'], head, replay_env, first_agent.encoder,
                                    at_fit, capacity, cell['environment']['horizon'])
            completed = []
            for step, frames in replays:
                point, agent, ref = at_fit[step]
                point_dir = destination / str(point)
                datasets, counts = batches(agent, frames, config)
                provenance = {'seed': seed, 'checkpoint': ref, 'checkpoint_decisions': point,
                              'fit_decisions': step, 'fit_id': agent.last_fit.fit_id,
                              'target_model_id': agent.last_fit.target_model_id,
                              'replay_first_id': frames[0][0].transition_id,
                              'replay_last_id': frames[-1][0].transition_id,
                              'by_action': counts}
                checked_write(point_dir / 'provenance.json', provenance)
                print(json.dumps(provenance), flush=True)
                if not (point_dir / 'published.json').is_file():
                    model_result = diagnostics(agent, agent.model, eval_env, bank, panel, config, stop)
                    checked_write(point_dir / 'published.json', model_result)
                for chain, condition in itertools.product(range(config['chains']), CONDITIONS):
                    if stop.requested:
                        raise InterruptedError('Diagnostic interrupted')
                    target = point_dir / f'{condition}-{chain}.json'
                    if target.is_file():
                        saved = checked_read(target)
                        if saved['provenance'] != digest(provenance):
                            raise ValueError('Stored chain provenance mismatch')
                        continue
                    fits = policies(agent, datasets[condition], config, point, chain)
                    outcomes = {budget: diagnostics(agent, model, eval_env, bank, panel, config, stop)
                                for budget, model in fits.items()}
                    checked_write(target, {'condition': condition, 'chain': chain,
                                           'provenance': digest(provenance), 'outcomes': outcomes})
                    print(json.dumps({'seed': seed, 'point': point, 'condition': condition,
                                      'chain': chain, 'returns': {
                                          name: mean(e['return'] for e in value['episodes'])
                                          for name, value in outcomes.items()}}), flush=True)
                completed.append(point)
            if completed != config['points']:
                raise ValueError('Coverage checkpoint set incomplete')
            checked_write(destination / 'complete.json', {'seed': seed, 'points': completed})
    finally:
        replay_env.close()
        eval_env.close()


def training_signature(row):
    fit = row['fit']
    return (row['interaction'], row['episode_id'], row['step_id'], row['episode_seed'], row['action'],
            row['raw_reward'], row['terminated'], row['truncated'], row['fit_id'],
            None if fit is None else (fit['target_model_id'], tuple(
                (action, data.get('data_hash')) for action, data in sorted(fit['actions'].items()))))


def run_control(output, historical, index, attempt_seconds):
    preflight = checked_read(output / 'preflight.json')
    run_cell(output, CONTROL, index, attempt_seconds)
    path = output / CONTROL
    manifest = load(path)
    cell = select_cell(manifest, str(index))
    history = checked_read(historical / 'campaign.json')
    if digest(history) != preflight['source_campaign']:
        raise ValueError('Cadence reference changed after preflight')
    old_cell = next(c for c in history['cells'] if c['seed'] == cell['seed'])
    folder, old_folder = path / cell['id'], historical / old_cell['id']
    status = checked_read(folder / 'status.json')
    old_status = checked_read(old_folder / 'status.json')
    if old_status['counter'] != 500000:
        raise ValueError('Unexpected cadence-reference endpoint')
    old = stream_rows(old_folder, old_status['log_head'])
    new = itertools.islice(stream_rows(folder, status['log_head']), old_status['counter'])
    count = 0
    first_difference = None
    for a, b in zip(old, new, strict=True):
        count += 1
        if first_difference is None and training_signature(a) != training_signature(b):
            first_difference = a['interaction']
    checked_write(folder / 'prefix_audit.json', {'decisions_compared': count,
                   'exact': first_difference is None, 'first_difference': first_difference})


def summarize_outcome(outcome):
    episodes, panel = outcome['episodes'], outcome['panel']
    return {phase: {
        'return': mean(e['return'] for e in episodes if e['phase'] == phase),
        'mean_fire_margin': mean(panel[phase]['fire_margin']),
        'fire_margin_positive_fraction': mean(m > 0 for m in panel[phase]['fire_margin']),
        'mean_top_margin': mean(panel[phase]['top_margin']),
        'initial_service': service_summary(
            [{'transitions': e['service'][:1]} for e in episodes if e['phase'] == phase],
            'reset' if phase == 'initial_serve' else 'life_lost') if phase != 'in_play' else None,
        'raw_episodes': [e for e in episodes if e['phase'] == phase],
    } for phase in PHASES}


def report(output, stability, config):
    preflight = checked_read(output / 'preflight.json')
    history = checked_read(stability / 'pilot' / 'report.json')
    control_path = output / CONTROL
    cells = load(control_path)['cells']
    control, diagnostic = {}, {}
    final = str(preflight['checkpoints'][-1])
    for cell in cells:
        seed = str(cell['seed'])
        folder = control_path / cell['id']
        summary = checked_read(folder / 'ablation_summary.json')
        by_point = {point: mean(r['raw_return'] for r in rows) for point, rows in summary['development'].items()}
        if set(map(int, by_point)) != set(preflight['checkpoints']):
            raise ValueError('Control evaluation calendar incomplete')
        final_rows = summary['development'][final]
        control[seed] = {'by_checkpoint': by_point, 'final': by_point[final],
                         'final_window_mean': mean(v for p, v in by_point.items() if int(p) >= 900000),
                         'initial_serve': service_summary(final_rows, 'reset'),
                         'after_life_loss': service_summary(final_rows, 'life_lost'),
                         'truncations': sum(r['truncated'] for r in final_rows),
                         'prefix_audit': checked_read(folder / 'prefix_audit.json'),
                         'paired_final_differences': {
                             arm: by_point[final] - row[seed]['final_mean_return']
                             for arm, row in history['per_seed'].items()}}
        root = output / 'coverage' / seed
        if checked_read(root / 'complete.json')['points'] != config['points']:
            raise ValueError('Coverage trial incomplete')
        diagnostic[seed] = {}
        for point in config['points']:
            point_dir = root / str(point)
            provenance = checked_read(point_dir / 'provenance.json')
            per_arm = {}
            for condition in CONDITIONS:
                chains = [checked_read(point_dir / f'{condition}-{i}.json')
                          for i in range(config['chains'])]
                if any(c['chain'] != i or c['condition'] != condition or c['provenance'] != digest(provenance)
                       for i, c in enumerate(chains)):
                    raise ValueError('Chain identity mismatch')
                per_arm[condition] = {}
                for budget in BUDGETS:
                    outcomes = [c['outcomes'][budget] for c in chains]
                    disagreements = {}
                    for phase in PHASES:
                        pairs = itertools.combinations(outcomes, 2)
                        disagreements[phase] = mean(
                            mean(a != b for a, b in zip(c['panel'][phase]['greedy'], d['panel'][phase]['greedy'], strict=True))
                            for c, d in pairs)
                    per_arm[condition][budget] = {'chains': [summarize_outcome(o) for o in outcomes],
                                                  'mean_pairwise_disagreement': disagreements}
            contrasts = {}
            for phase in PHASES:
                def arm_mean(arm, budget):
                    return mean(c[phase]['return'] for c in per_arm[arm][budget]['chains'])
                contrasts[phase] = {
                    'coverage_at_standard': arm_mean('phase_coverage', 'standard') - arm_mean('original', 'standard'),
                    'coverage_at_extended': arm_mean('phase_coverage', 'extended') - arm_mean('original', 'extended'),
                    'longer_original': arm_mean('original', 'extended') - arm_mean('original', 'standard'),
                    'longer_with_coverage': arm_mean('phase_coverage', 'extended') - arm_mean('phase_coverage', 'standard'),
                }
            diagnostic[seed][str(point)] = {'provenance': provenance, 'arms': per_arm,
                                          'published': summarize_outcome(checked_read(point_dir / 'published.json')),
                                          'paired_return_contrasts': contrasts}
    checked_write(output / 'report.json', {'matched_control': control, 'fixed_checkpoint_diagnostics': diagnostic,
                   'control_mean_final': mean(row['final'] for row in control.values()),
                   'control_mean_final_window': mean(row['final_window_mean'] for row in control.values()),
                   'reserved_test_episodes': 0})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'coverage', 'control', 'report'))
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--historical', required=True, type=Path)
    parser.add_argument('--stability', required=True, type=Path)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--stability-config', required=True, type=Path)
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--attempt-seconds', type=int, default=85000)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.stage == 'prepare':
        launch_experiment(args.output, args.historical, args.stability, args.stability_config, config)
    elif args.stage == 'coverage':
        coverage(args.output, args.historical, config, args.index)
    elif args.stage == 'control':
        run_control(args.output, args.historical, args.index, args.attempt_seconds)
    else:
        report(args.output, args.stability, config)


if __name__ == '__main__':
    main()
