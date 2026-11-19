"""Run paired RRTL-ANOVA follow-ups on retained CPU-cluster checkpoints."""
import argparse
import copy
import gzip
import json
import pickle
from dataclasses import asdict, replace
from functools import partial
from pathlib import Path
from statistics import mean, median

import numpy as np

from bayesian_rrtl.online.protocol import runtime, source_hashes
from bayesian_rrtl.online.session import OnlineSession
from bayesian_rrtl.revision_diagnostics import (
    make_start_bank, rollout, validation_probes,
)
from bayesian_rrtl.rrtl_followup import (
    FollowupAgent, PHASES, double_targets, make_phase_bank, policy_on_start,
)
from bayesian_rrtl.stable_revision import variant_config
from bayesian_rrtl.tree_revision import RevisionConfig
from experiments import diagnose_rrtl_improvements as original
from experiments import diagnose_rrtl_revision as common
from experiments.compare_pong_progressive import StopRequest, default_factory
from experiments.pong_analysis_runtime import (
    WorkUnit, bytes_hash, checked_read, checked_write, digest, unit_lock,
)

SCHEMA = 'rrtl_anova_followup_v1'
FORK_SCHEMA = 'rrtl_anova_followup_fork_v1'
ARMS = ('baseline', 'slow_td', 'frozen_td', 'frozen_slow_td', 'refit_only', 'refit_damped', 'guarded')
REFERENCES = {'baseline', 'slow_td', 'refit_only'}
SOURCE_FIELDS = ('interaction', 'transition_id', 'episode_id', 'step_id', 'episode_seed',
                 'epsilon', 'action', 'reward', 'raw_reward', 'terminated', 'truncated', 'td')


def read_campaign(output):
    manifest = checked_read(output / 'manifest.json')
    if (manifest['schema'] != SCHEMA or manifest['sources'] != source_hashes()
            or manifest['runtime'] != runtime('corrected_common')):
        raise ValueError('Follow-up source or runtime changed')
    return manifest, common.blob_read(output, manifest['start_bank'])


def reference_dir(reference, parent, arm):
    return Path(reference) / 'continuations' / 'anova' / str(parent['seed']) / arm


def prepare(config_path, parents, reference, output):
    config = json.loads(config_path.read_text())
    if (config['modes'] != list(ARMS) or parents.resolve() == output.resolve()
            or reference.resolve() == output.resolve() or parents.resolve() == reference.resolve()):
        raise ValueError('Invalid isolated follow-up design')
    schedule = config['schedule']
    if (set(schedule) != set(ARMS) or any(n < 100000 or n % config['evaluation_interval'] for n in schedule.values())
            or config['continuation_decisions'] != max(schedule.values())
            or config['prune_points'] != sorted(set(config['prune_points']))
            or not config['prune_points'] or config['prune_points'][0] != 0):
        raise ValueError('Invalid follow-up milestones')

    previous = checked_read(reference / 'manifest.json')
    refit_bank = common.blob_read(reference, previous['validation_bank'])
    refit_ram = {start['ram_sha256'] for start in refit_bank['starts']}

    def extend(manifest, work):
        if previous['schema'] != 'rrtl_anova_confirmation_v1' or previous['runtime'] != manifest['runtime']:
            raise ValueError('Expected the prior ANOVA confirmation campaign')
        manifest['reference_campaign'] = str(reference.resolve())
        manifest['reference_digest'] = digest(previous)
        manifest['references'] = {}
        previous_points = previous['config']['evaluation_interval']
        if previous_points != config['evaluation_interval']:
            raise ValueError('Reference cadence differs')
        for parent in manifest['parents']:
            seed = str(parent['seed'])
            old_parent = next(row for row in previous['parents'] if row['seed'] == parent['seed'])
            if old_parent['checkpoint'] != parent['checkpoint'] or old_parent['cell'] != parent['cell']:
                raise ValueError(f'Reference checkpoint differs for {seed}')
            arms = {}
            for arm in sorted(REFERENCES):
                if previous['config']['variants'][arm] != asdict(variant_config(arm)):
                    raise ValueError(f'Reused {arm} settings differ')
                folder = reference_dir(reference, parent, arm)
                status = checked_read(folder / 'status.json')
                points = range(0, schedule[arm] + 1, previous_points)
                expected = common.fork_identity(previous, old_parent, arm)
                if status['identity'] != expected or status['state'] != 'completed':
                    raise ValueError(f'Invalid reference status {seed}/{arm}')
                chosen = {str(point): status['checkpoints'][str(point)] for point in points}
                if any(not (folder / ref['path']).is_file() for ref in chosen.values()):
                    raise FileNotFoundError(f'Missing retained reference {seed}/{arm}')
                arms[arm] = {'identity': expected, 'checkpoints': chosen}
            manifest['references'][seed] = arms
        main_bank = common.blob_read(work, manifest['start_bank'])
        probes = common.blob_read(work, manifest['validation_bank'])
        manifest['refit_validation_bank'] = common.blob_write(work, 'refit_validation_bank', refit_bank)
        manifest['refit_probe_settings'] = {
            name: previous['config'][name] for name in ('probe_seed', 'probe_blocks', 'probe_horizon')}
        if previous['config']['revision'] != config['revision']:
            raise ValueError('Damped refit must use the full-refit maintenance settings')
        forbidden = {start['ram_sha256'] for bank in (main_bank, probes) for start in bank['starts']} | refit_ram
        manifest['return_banks'] = {}
        environment = default_factory(manifest['parents'][0]['cell'])()
        try:
            for point in range(config['evaluation_interval'], schedule['guarded'] + 1,
                               config['evaluation_interval']):
                guard_bank = make_start_bank(environment, count=config['return_starts'],
                                             seed=config['return_start_seed'] + point, excluded_ram=forbidden)
                original.verify_bank(environment, guard_bank)
                forbidden.update(start['ram_sha256'] for start in guard_bank['starts'])
                manifest['return_banks'][str(point)] = common.blob_write(work, f'return_bank_{point}', guard_bank)
        finally:
            environment.close()
        manifest['scope'] = 'Exploratory follow-up on reused ANOVA parents; reserved test excluded'
        return manifest

    return common.prepare(config_path, parents, parents, output, expected_modes=ARMS,
                          manifest_schema=SCHEMA, existing_loader=read_campaign,
                          extend_manifest=extend, criteria=('anova',),
                          start_bank_factory=partial(make_phase_bank, excluded_ram=refit_ram))


def variant_for(base, config, arm, seed):
    revision = RevisionConfig(**{**config['revision'], 'seed': config['revision']['seed'] + seed})
    preset = arm
    if arm in ('frozen_td', 'frozen_slow_td'):
        revision = replace(revision, mode='frozen')
        preset = 'baseline' if arm == 'frozen_td' else 'slow_td'
    if arm == 'refit_damped':
        preset = 'refit_only'
    damping = config['refit_fraction'] if arm == 'refit_damped' else 1.0
    return FollowupAgent.fork(base, revision, variant_config(preset), refit_fraction=damping)


def new_session(parent, environment, config, arm):
    saved = common.parent_payload(parent)['session']
    session = OnlineSession.restore(saved, parent['cell'], environment)
    session.agent = variant_for(session.agent, config, arm, parent['seed'])
    return session


def evaluation_point(folder, manifest, bank, agent, environment, identity, stop):
    """Evaluate a frozen model, persisting each paired start independently."""
    config = manifest['config']
    for start in bank['starts']:
        if stop.requested:
            return False
        target = folder / str(start['id'])
        seed = config['evaluation_seed'] + start['id']
        start_identity = {**identity, 'bank': manifest['start_bank'], 'start': start['id'],
                          'phase': start['phase'], 'seed': seed, 'horizon': config['evaluation_horizon']}
        with unit_lock(target):
            work = WorkUnit(target, start_identity)
            saved = work.load()
            if saved and saved['status'] == 'completed':
                continue
            result = rollout(environment, agent, start, horizon=config['evaluation_horizon'],
                             seed=seed, serve_window=config['serve_window'])
            result['start_policy'] = policy_on_start(agent, start['state'])
            work.commit('completed', result=result)
    return True


def reuse_reference(output, manifest, bank, parent, arm, stop):
    reference = Path(manifest['reference_campaign'])
    if digest(checked_read(reference / 'manifest.json')) != manifest['reference_digest']:
        raise ValueError('Earlier reference manifest changed')
    folder = reference_dir(reference, parent, arm)
    verified = manifest['references'][str(parent['seed'])][arm]
    environment = default_factory(parent['cell'])()
    try:
        for point, ref in sorted(verified['checkpoints'].items(), key=lambda item: int(item[0])):
            saved = common.blob_read(folder, ref)
            if saved['identity'] != verified['identity'] or saved['counter'] != parent['counter'] + int(point):
                raise ValueError('Wrong reused checkpoint')
            identity = {**common.fork_identity(manifest, parent, arm), 'checkpoint': ref, 'point': int(point)}
            agent = saved['session']['collector']['agent']
            if agent.interactions != saved['counter']:
                raise ValueError('Reused learner counter differs')
            where = output / 'results' / str(parent['seed']) / arm
            if not evaluation_point(where / point, manifest, bank, agent, environment, identity, stop):
                return False
        checked_write(where / 'complete.json', {'identity': verified['identity'], 'checkpoints': verified['checkpoints']})
    finally:
        environment.close()
    return True


def maintain(session, parent, manifest, output, folder, probe_env, probe_bank, point):
    """Save the two exact same-time sessions around each accepted protected prune."""
    config = manifest['config']
    probes = manifest['refit_probe_settings'] if session.agent.improvement.repair == 'refit' else config
    validation = validation_probes(probe_env, probe_bank, seed=probes['probe_seed'] + point,
                                   blocks=probes['probe_blocks'], horizon=probes['probe_horizon'])
    if session.agent.improvement.repair == 'guarded':
        bank = common.blob_read(output, manifest['return_banks'][str(point)])
        guard = original.return_validator(probe_env, bank, config, point, parent['seed'])
        before = copy.deepcopy(session.state_dict())
    else:
        guard = None
        before = None
    reviews = session.agent.maintain(validation, guard)
    record = {'counter': session.interactions, 'reviews': reviews}
    accepted = [event for event in reviews if event['kind'] == 'prune_review' and event['gate']['accepted']]
    if accepted:
        record['prune_pair'] = {
            'before': common.blob_write(folder, f'pre_prune_{point}', before),
            'after': common.blob_write(folder, f'post_prune_{point}', session.state_dict()),
        }
    checked_write(folder / 'reviews' / f'{point}.json', record)


def train(output, manifest, bank, parent, arm, stop):
    config = manifest['config']
    identity = common.fork_identity(manifest, parent, arm)
    folder = output / 'training' / str(parent['seed']) / arm
    destination = output / 'results' / str(parent['seed']) / arm
    cadence = config['evaluation_interval']
    end = config['schedule'][arm]
    probe_key = 'refit_validation_bank' if arm == 'refit_damped' else 'validation_bank'
    probe_bank = common.blob_read(output, manifest[probe_key])
    with unit_lock(folder):
        environment = default_factory(parent['cell'])()
        probe_env = default_factory(parent['cell'])()
        try:
            if (folder / 'status.json').exists():
                status = checked_read(folder / 'status.json')
                session = common.restore_fork(folder, status, identity, environment,
                                              agent_class=FollowupAgent, schema=FORK_SCHEMA)
            else:
                session = new_session(parent, environment, config, arm)
                status = common.save_fork(folder, identity, session, None, schema=FORK_SCHEMA)
            while True:
                point = session.interactions - parent['counter']
                if point % cadence == 0:
                    ref = status['checkpoints'][str(point)]
                    if not evaluation_point(destination / str(point), manifest, bank, session.agent, probe_env,
                                            {**identity, 'checkpoint': ref, 'point': point}, stop):
                        return False
                    print(json.dumps({'seed': parent['seed'], 'arm': arm, 'counter': session.interactions,
                                      'metrics': session.agent.revision_metrics()}), flush=True)
                if point == end:
                    checked_write(folder / 'status.json', {**status, 'state': 'completed'})
                    checked_write(destination / 'complete.json', {'identity': identity, 'checkpoints': status['checkpoints']})
                    return True
                next_point = (point // cadence + 1) * cadence
                while session.interactions < parent['counter'] + next_point:
                    if stop.requested:
                        # The prior boundary is committed, with deterministic replay of this interval.
                        return False
                    row = session.step()
                    row['diagnostic'] = session.agent.last_detail
                if session.agent.improvement.repair != 'none':
                    maintain(session, parent, manifest, output, folder, probe_env, probe_bank, next_point)
                status = common.save_fork(folder, identity, session, status, schema=FORK_SCHEMA)
        finally:
            environment.close()
            probe_env.close()


def audit_double(output, manifest, parent, stop):
    """Reconstruct a retained prefix and compare the three backups at a 5k lag."""
    config = manifest['config']
    lag = config['double_lag']
    destination = output / 'double' / str(parent['seed'])
    source = reference_dir(manifest['reference_campaign'], parent, 'baseline')
    status = checked_read(source / 'status.json')
    first = status['traces'][0]
    raw = (source / first['path']).read_bytes()
    if (bytes_hash(raw) != first['sha256'] or first['start'] != parent['counter'] + 1
            or first['end'] < parent['counter'] + lag):
        raise ValueError('Prior baseline trace disagrees with Double audit range')
    rows = gzip.decompress(raw).splitlines()[:lag]
    environment = default_factory(parent['cell'])()
    with unit_lock(destination):
        try:
            session = new_session(parent, environment, config, 'baseline')
            frozen = copy.deepcopy(session.agent.root_nodes)
            for i, line in enumerate(rows, 1):
                if stop.requested:
                    return False
                actual = session.step()
                expected = json.loads(line)
                if actual['interaction'] != parent['counter'] + i or any(
                        actual[field] != expected[field] for field in SOURCE_FIELDS):
                    raise ValueError(f'Exact replay mismatch at transition {i}')
            records = list(session.agent.buffer)
            if len(records) != lag:
                raise ValueError('Double audit buffer lost replay records')
            online = session.agent.root_nodes
            panel = list(double_targets(records, online, frozen, gamma=session.agent.config.gamma,
                                       bootstrap_on_truncation=session.agent.config.bootstrap_on_truncation))
            bound = session.agent.config.gamma
            if any(row['double'] > row['frozen_max'] + 1e-12 for row in panel):
                raise ValueError('Double exceeds the frozen maximum')
            checkpoint = common.blob_write(destination, 'same_panel', {
                'records': records, 'frozen_roots': frozen, 'online_roots': online,
            })
            effects = {}
            live = [row for row in panel if row['mask']]
            for a, b in (('frozen_max', 'online_max'), ('double', 'frozen_max'), ('double', 'online_max')):
                difference = np.array([row[a] - row[b] for row in live])
                effects[f'{a}_minus_{b}'] = {
                    'mean': float(difference.mean()), 'mean_abs': float(np.abs(difference).mean()),
                    'quantiles_05_50_95': np.quantile(difference, [0.05, 0.5, 0.95]).tolist(),
                    'different': int(np.count_nonzero(difference)),
                }
            checked_write(destination / 'results.json', {
                'seed': parent['seed'], 'parent': parent['checkpoint'], 'retained_prefix': first,
                'frozen_at': parent['counter'], 'online_at': session.interactions, 'gamma': bound,
                'panel': checkpoint, 'transitions': len(panel), 'bootstrap_rows': len(live),
                'disjoint_greedy': sum(not set(row['online_greedy']) & set(row['frozen_greedy']) for row in live),
                'effects': effects, 'targets': panel,
            })
        finally:
            environment.close()
    return True


def restore_prune_session(saved, environment):
    if saved['runtime_environment'] != environment.runtime_manifest():
        raise ValueError('Prune-pair environment contract changed')
    session = OnlineSession.__new__(OnlineSession)
    vars(session).update(saved['collector'])
    session.environment, session.pending_rows = environment, []
    session.agent.validate_revision()
    environment.restore(saved['environment'])
    return session


def pruning_pairs(output, manifest, bank, parent, stop):
    config = manifest['config']
    seed = parent['seed']
    folder = output / 'training' / str(seed) / 'guarded'
    status = checked_read(folder / 'status.json')
    if status['state'] != 'completed' or status['identity'] != common.fork_identity(manifest, parent, 'guarded'):
        raise ValueError('Guarded run has not completed')
    destination = output / 'pruning' / str(seed)
    with unit_lock(destination):
        environment = default_factory(parent['cell'])()
        probe_env = default_factory(parent['cell'])()
        try:
            completed = []
            for point in range(config['evaluation_interval'], config['schedule']['guarded'] + 1,
                               config['evaluation_interval']):
                record = checked_read(folder / 'reviews' / f'{point}.json')
                if 'prune_pair' not in record:
                    continue
                refs = record['prune_pair']
                branches = {arm: common.blob_read(folder, ref) for arm, ref in refs.items()}
                before, after = branches['before'], branches['after']
                if (pickle.dumps(before['environment'], protocol=4) != pickle.dumps(after['environment'], protocol=4)
                        or before['collector']['state'] != after['collector']['state']
                        or any(digest(before['collector'][rng].bit_generator.state) !=
                               digest(after['collector'][rng].bit_generator.state)
                               for rng in ('environment_rng', 'exploration_rng'))):
                    raise ValueError('Prune pair must share the same environment and collection RNGs')
                for arm, saved in branches.items():
                    session = restore_prune_session(saved, environment)
                    if session.interactions != record['counter']:
                        raise ValueError('Prune pair counter differs')
                    for offset in config['prune_points']:
                        while session.interactions < record['counter'] + offset:
                            if stop.requested:
                                return False
                            session.step()
                            session.pending_rows.clear()
                        identity = {'seed': seed, 'prune_counter': record['counter'], 'branch': arm,
                                    'snapshot': refs[arm], 'offset': offset}
                        where = destination / str(record['counter']) / arm / str(offset)
                        if not evaluation_point(where, manifest, bank, session.agent, probe_env, identity, stop):
                            return False
                        checked_write(where / 'metrics.json', session.agent.revision_metrics())
                completed.append(record['counter'])
            checked_write(destination / 'complete.json', {
                'seed': seed, 'prune_counters': completed, 'offsets': config['prune_points'],
            })
        finally:
            environment.close()
            probe_env.close()
    return True


def score_point(folder, bank, serve_window):
    by_phase = {phase: [] for phase in PHASES}
    serve = {phase: {'fired': 0, 'missed': 0, 'censored': 0}
             for phase in ('initial_serve', 'after_life_loss')}
    actions = []
    for start in bank['starts']:
        location = folder / str(start['id'])
        identity = checked_read(location / 'identity.json')
        saved = checked_read(location / 'state.json')
        result = saved['result']
        if (saved['status'] != 'completed' or saved['identity_sha256'] != digest(identity)
                or result['physical_hash'] != start['physical_hash']
                or identity['phase'] != start['phase'] or result['start'] != start['id']):
            raise ValueError('Incomplete or mismatched follow-up evaluation')
        by_phase[start['phase']].append(result['raw_return'])
        actions.append(result['start_policy'])
        for opportunity in result['serve_opportunities']:
            counts = serve[opportunity['phase']]
            if opportunity['first_fire'] is not None:
                counts['fired'] += 1
            elif opportunity['observed'] == serve_window:
                counts['missed'] += 1
            else:
                counts['censored'] += 1
    return {'return': mean(mean(scores) for scores in by_phase.values()),
            'by_phase': {phase: mean(scores) for phase, scores in by_phase.items()},
            'serve': serve, 'policy': actions, 'mean_start_margin': mean(row['greedy_margin'] for row in actions)}


def report(output, manifest, bank):
    config = manifest['config']
    seed_reports = {}
    for parent in manifest['parents']:
        seed = str(parent['seed'])
        arms = {}
        for arm, end in config['schedule'].items():
            location = output / 'results' / seed / arm
            status = checked_read(location / 'complete.json')
            points = range(0, end + 1, config['evaluation_interval'])
            if set(status['checkpoints']) != {str(point) for point in points}:
                raise ValueError(f'Incomplete result schedule {seed}/{arm}')
            curve = {str(point): score_point(location / str(point), bank, config['serve_window'])
                     for point in points}
            origin = curve['0']
            final = curve[str(end)]
            tail = [point['return'] for offset, point in curve.items()
                    if int(offset) >= end - config['final_window']]
            arms[arm] = {'curve': curve, 'final': final['return'],
                         'final_window_mean': mean(tail), 'final_minus_parent': final['return'] - origin['return'],
                         'final_changed_start_argmax': sum(
                             set(a['greedy']) != set(b['greedy']) for a, b in zip(origin['policy'], final['policy']))}
        prune_dir = output / 'pruning' / seed
        summary = checked_read(prune_dir / 'complete.json')
        pairs = {}
        for counter in summary['prune_counters']:
            pairs[str(counter)] = {}
            for offset in summary['offsets']:
                scores = {arm: score_point(prune_dir / str(counter) / arm / str(offset), bank, config['serve_window'])
                          for arm in ('before', 'after')}
                pairs[str(counter)][str(offset)] = {**scores, 'paired_delta': scores['after']['return'] - scores['before']['return']}
        offline = checked_read(output / 'double' / seed / 'results.json')
        seed_reports[seed] = {'arms': arms, 'pruning': pairs,
                              'double': {k: v for k, v in offline.items() if k != 'targets'}}
    factorial = {}
    refit = {}
    for seed, results in seed_reports.items():
        arms = results['arms']
        scores = {arm: arms[arm]['curve']['100000']['return']
                  for arm in ('baseline', 'slow_td', 'frozen_td', 'frozen_slow_td')}
        live = scores['slow_td'] - scores['baseline']
        fixed = scores['frozen_slow_td'] - scores['frozen_td']
        factorial[seed] = {'at_600k': scores, 'slow_effect_growth': live,
                           'slow_effect_frozen': fixed, 'interaction': live - fixed}
        refit[seed] = {'final_delta': arms['refit_damped']['final'] - arms['refit_only']['final'],
                       'final_window_delta': arms['refit_damped']['final_window_mean'] - arms['refit_only']['final_window_mean']}
    checked_write(output / 'report.json', {
        'campaign': digest(manifest), 'seed_reports': seed_reports, 'factorial': factorial, 'refit': refit,
        'summary': {arm: {
            'final': mean(r['arms'][arm]['final'] for r in seed_reports.values()),
            'median_final': median(r['arms'][arm]['final'] for r in seed_reports.values()),
            'final_window': mean(r['arms'][arm]['final_window_mean'] for r in seed_reports.values()),
        } for arm in ARMS},
    })


def smoke(output, manifest, bank):
    """Exercise ALE, all new arm configurations, and exact session continuation."""
    parent = manifest['parents'][0]
    config = manifest['config']
    environment = default_factory(parent['cell'])()
    probe = default_factory(parent['cell'])()
    try:
        assertions = {}
        for arm in (mode for mode in ARMS if mode not in REFERENCES):
            session = new_session(parent, environment, config, arm)
            expected_splits = session.agent.n_splits
            for _ in range(64):
                session.step()
            result = rollout(probe, session.agent, bank['starts'][0], horizon=64,
                             seed=config['evaluation_seed'], serve_window=config['serve_window'])
            if session.agent.revision.mode == 'frozen' and session.agent.n_splits != expected_splits:
                raise ValueError('Frozen arm grew during smoke')
            saved = copy.deepcopy(session.state_dict())
            next_original = session.step()
            fork = restore_prune_session(saved, environment)
            next_restored = fork.step()
            if any(next_restored[field] != next_original[field] for field in SOURCE_FIELDS):
                raise ValueError('Follow-up exact resume failed')
            assertions[arm] = {'counter': fork.interactions, 'phase': bank['starts'][0]['phase'],
                               'steps': result['steps']}
        checked_write(output / 'preflight.json', {'source': digest(manifest), 'smoke': assertions})
    finally:
        environment.close()
        probe.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'worker', 'double', 'pruning', 'report'))
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--parents', type=Path)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--index', type=int, default=0)
    args = parser.parse_args()
    if args.stage == 'prepare':
        prepare(args.config, args.parents, args.reference, args.output)
        manifest, bank = read_campaign(args.output)
        smoke(args.output, manifest, bank)
        return
    manifest, bank = read_campaign(args.output)
    if checked_read(args.output / 'preflight.json')['source'] != digest(manifest):
        raise ValueError('Missing successful preflight for this manifest')
    if args.stage == 'report':
        report(args.output, manifest, bank)
        return
    with StopRequest() as stop:
        if args.stage == 'worker':
            parent_index, arm_index = divmod(args.index, len(ARMS))
            parent, arm = manifest['parents'][parent_index], ARMS[arm_index]
            if arm in REFERENCES:
                ok = reuse_reference(args.output, manifest, bank, parent, arm, stop)
            else:
                ok = train(args.output, manifest, bank, parent, arm, stop)
        else:
            parent = manifest['parents'][args.index]
            if args.stage == 'double':
                ok = audit_double(args.output, manifest, parent, stop)
            else:
                ok = pruning_pairs(args.output, manifest, bank, parent, stop)
    if not ok:
        raise SystemExit(3)


if __name__ == '__main__':
    main()
