"""Isolated Breakout revision experiment: trusted parent forks, never migration.

Stages: prepare, frozen, continue, report. All original campaigns are read-only.
Python/ALE stages must run on a compute node on the target cluster.
"""
import argparse
import copy
from dataclasses import asdict
import gzip
import json
from pathlib import Path
import pickle
import time

import numpy as np

from bayesian_rrtl.baseline_checkpoint import _validate_trees
from bayesian_rrtl.online.protocol import runtime, source_hashes
from bayesian_rrtl.online.session import OnlineSession
from bayesian_rrtl.online.storage import checkpoint_metadata, load_checkpoint, verified_bytes
from bayesian_rrtl.revision_diagnostics import make_start_bank, rollout, validation_probes, verify_start
from bayesian_rrtl.revision_diagnostics import physical_hash, ram_hash
from bayesian_rrtl.baseline_adapter import normalize_facts
from bayesian_rrtl.tree_revision import MODES, RevisionAgent, RevisionConfig
from experiments.compare_pong_progressive import StopRequest, default_factory
from experiments.pong_analysis_runtime import (WorkUnit, atomic_bytes, bytes_hash, checked_read,
                                              checked_write, digest, unit_lock)

SCHEMA = 'rrtl_revision_campaign_v1'


def parent_payload(entry, *, verify_history=False):
    """Explicit new-experiment fork, not an original-protocol resume.

    Exact old source/runtime bindings remain recorded. The model and its metadata
    are always checked; preparation additionally checks the committed log chain.
    OnlineSession.restore separately enforces exact emulator/detector manifests.
    """
    folder = Path(entry['folder'])
    manifest = checked_read(Path(entry['campaign'])/'campaign.json')
    if digest(manifest) != entry['campaign_sha256']:
        raise ValueError('Parent campaign changed')
    cell = next(c for c in manifest['cells'] if c['id'] == entry['cell']['id'])
    if cell != entry['cell'] or cell['learner'] != 'corrected_common':
        raise ValueError('Parent cell mismatch')
    expected = {'campaign': digest(manifest), 'cell': cell['hash'],
                'learner': cell['learner'], 'runtime': manifest['runtime']}
    if manifest['runtime'] != runtime('corrected_common'):
        raise ValueError('Fork requires the pinned parent numeric runtime')
    if verify_history:
        payload, _ = load_checkpoint(folder, entry['checkpoint'], expected)
    else:
        meta = checkpoint_metadata(folder, entry['checkpoint'])
        if meta['binding'] != expected:
            raise ValueError('Parent checkpoint binding mismatch')
        payload = pickle.loads(verified_bytes(folder, entry['checkpoint']))
        if payload['binding'] != expected or payload['counter'] != meta['counter'] or payload['log_head'] != meta['log_head']:
            raise ValueError('Parent checkpoint frontier mismatch')
    if payload['counter'] != entry['counter']:
        raise ValueError('Wrong parent counter')
    agent = payload['session']['collector']['agent']
    if agent.interactions != payload['counter']:
        raise ValueError('Parent model/counter mismatch')
    _validate_trees(agent)
    return payload


def blob_write(folder, name, value):
    raw = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    sha = bytes_hash(raw)
    path = Path(folder)/f'{name}-{sha}.pkl'
    if not path.exists():
        try:
            atomic_bytes(path, raw, replace=False)
        except FileExistsError:
            # Identical shared holdouts may be published by concurrent groups.
            if bytes_hash(path.read_bytes()) != sha:
                raise ValueError('Conflicting concurrent diagnostic blob')
    return {'path': path.name, 'sha256': sha}


def blob_read(folder, ref):
    if Path(ref['path']).name != ref['path']:
        raise ValueError('Invalid blob path')
    raw = (Path(folder)/ref['path']).read_bytes()
    if bytes_hash(raw) != ref['sha256']:
        raise ValueError('Diagnostic blob hash mismatch')
    return pickle.loads(raw)


def load_campaign(output):
    output = Path(output)
    manifest = checked_read(output/'manifest.json')
    if manifest['schema'] != SCHEMA or manifest['sources'] != source_hashes() or manifest['runtime'] != runtime('corrected_common'):
        raise ValueError('Diagnostic campaign code/runtime mismatch')
    bank = blob_read(output, manifest['start_bank'])
    return manifest, bank


def prepare(config_path, historical, anova, output, *, expected_modes=MODES,
            manifest_schema=SCHEMA, existing_loader=None, extend_manifest=None,
            criteria=('historical_f', 'anova'), start_bank_factory=make_start_bank):
    config = json.loads(Path(config_path).read_text())
    if config['modes'] != list(expected_modes) or config['parent_counter'] not in config['checkpoints']:
        raise ValueError('Explicit complete revision design required')
    if config['continuation_decisions'] % config['evaluation_interval'] or config['evaluation_interval'] % config['save_interval']:
        raise ValueError('Checkpoint/evaluation cadence mismatch')
    RevisionConfig(**config['revision'])
    output = Path(output)
    with unit_lock(output):
        if (output/'manifest.json').exists():
            manifest, _ = (existing_loader or load_campaign)(output)
            if manifest['config'] != config:
                raise ValueError('Existing diagnostic design differs')
            return manifest
        parents = []
        if not criteria or len(set(criteria)) != len(criteria) or set(criteria)-{'historical_f','anova'}:
            raise ValueError('Explicit supported criteria required')
        for criterion in criteria:
            campaign = {'historical_f':historical, 'anova':anova}[criterion]
            campaign = Path(campaign).resolve()
            manifest = checked_read(campaign/'campaign.json')
            for seed in config['seeds']:
                cell = next(c for c in manifest['cells'] if c['seed'] == seed and c['game'] == 'Breakout' and c['learner'] == 'corrected_common')
                if cell['estimator'].get('split_test', 'historical_f') != criterion:
                    raise ValueError('Wrong parent criterion')
                folder = campaign/cell['id']
                for counter in config['checkpoints']:
                    paths = list((folder/'checkpoints').glob(f'{counter}-*.pkl'))
                    if len(paths) != 1:
                        raise ValueError(f'Expected exactly one retained parent at {criterion}/{seed}/{counter}')
                    ref = {'path': str(paths[0].relative_to(folder)), 'sha256': paths[0].stem.split('-',1)[1]}
                    entry = {'criterion': criterion, 'seed': seed, 'counter': counter, 'checkpoint': ref,
                        'campaign': str(campaign), 'campaign_sha256': digest(manifest), 'folder': str(folder), 'cell': cell}
                    parent_payload(entry, verify_history=True)
                    parents.append(entry)
                    print(json.dumps({'verified_parent': [criterion, seed, counter]}), flush=True)
        first = parents[0]
        for parent in parents:
            if parent['cell']['environment'] != first['cell']['environment']:
                raise ValueError('Parent environments differ')
        env = default_factory(first['cell'])()
        try:
            bank = start_bank_factory(env, count=config['starts'], seed=config['start_seed'])
            validation_bank = make_start_bank(env, count=config.get('probe_starts', config['starts']), seed=config['validation_start_seed'],
                excluded_ram=[s['ram_sha256'] for s in bank['starts']])
            # Real exact-state continuity check, including detector temporal memory.
            for start in [*bank['starts'], *validation_bank['starts']]:
                state = env.reset(seed=start['reset_seed'])
                for action in start['prefix_actions']:
                    state, _, _, terminal, truncated = env.step(action)
                    if terminal or truncated: raise ValueError('Prefix terminated before its captured start')
                if (normalize_facts(state) != start['state'] or physical_hash(env) != start['physical_hash']
                        or ram_hash(env) != start['ram_sha256']):
                    raise ValueError('Logged prefix does not reproduce its live captured start')
                a = env.step(1)
                reference_hash = physical_hash(env)
                verify_start(env, start)
                b = env.step(1)
                if a != b or physical_hash(env) != reference_hash:
                    raise ValueError('Full snapshot continuation is not reproducible')
        finally:
            env.close()
        bank_ref = blob_write(output, 'start_bank', bank)
        validation_ref = blob_write(output, 'validation_bank', validation_bank)
        result = {'schema': manifest_schema, 'config': config, 'parents': parents,
            'groups': [{'criterion': criterion, 'seed': seed} for criterion in criteria for seed in config['seeds']],
            'sources': source_hashes(), 'runtime': runtime('corrected_common'), 'start_bank': bank_ref,
            'validation_bank': validation_ref,
            'start_diversity': {k: bank[k] for k in ('unique_physical','unique_encoded','unique_ram')},
            'scope': 'new diagnostic experiment; original parents/read-only, no reserved test',
            'parent_histories_verified': True}
        if extend_manifest is not None:
            result = extend_manifest(result, output)
        checked_write(output/'manifest.json', result, replace=False)
        checked_write(output/'starts.json', [{k:v for k,v in start.items() if k not in ('state','environment')}
                                            for start in bank['starts']])
        checked_write(output/'validation_starts.json', [{k:v for k,v in start.items() if k not in ('state','environment')}
                                                       for start in validation_bank['starts']])
        return result


def evaluation_unit(folder, identity, environment, agent, start, epsilon, horizon, seed):
    with unit_lock(folder):
        work = WorkUnit(folder, identity)
        saved = work.load()
        if saved and saved['status'] == 'completed':
            return saved['result']
        result = rollout(environment, agent, start, epsilon=epsilon, horizon=horizon, seed=seed)
        work.commit('completed', result=result)
        return result


def verified_evaluation(path, expected):
    saved = checked_read(path)
    identity = checked_read(Path(path).with_name('identity.json'))
    if identity != expected or saved['identity_sha256'] != digest(identity):
        raise ValueError('Evaluation identity/checkpoint mismatch')
    result = saved['result']
    if (result['start'] != expected['start'] or result['epsilon'] != expected['epsilon']
            or result['horizon'] != expected['horizon'] or result['seed'] != expected['evaluation_seed']):
        raise ValueError('Evaluation result settings mismatch')
    return saved


def frozen_group(output, group_index, stop):
    output = Path(output); manifest, bank = load_campaign(output)
    config = manifest['config']; group = manifest['groups'][group_index]
    parents = [p for p in manifest['parents'] if (p['criterion'],p['seed']) == (group['criterion'],group['seed'])]
    env = default_factory(parents[0]['cell'])()
    try:
        for parent in parents:
            payload = parent_payload(parent)
            agent = payload['session']['collector']['agent']
            before = pickle.dumps(agent, protocol=4)
            for epsilon in config['evaluation_epsilons']:
                for start in bank['starts']:
                    if stop.requested:
                        return False
                    identity = {'campaign': digest(manifest), 'parent': parent['checkpoint'],
                        'start_bank': manifest['start_bank'], 'start': start['id'], 'epsilon': epsilon,
                        'evaluation_seed': config['evaluation_seed']+start['id'],
                        'horizon': config['evaluation_horizon'], 'policy': 'frozen_epsilon_greedy_uniform_ties'}
                    folder = output/'frozen'/parent['criterion']/str(parent['seed'])/str(parent['counter'])/str(epsilon)/str(start['id'])
                    evaluation_unit(folder, identity, env, agent, start, epsilon, config['evaluation_horizon'],
                                    config['evaluation_seed']+start['id'])
            if pickle.dumps(agent, protocol=4) != before:
                raise ValueError('Frozen evaluation modified the model')
            print(json.dumps({'frozen_completed': [parent['criterion'],parent['seed'],parent['counter']]}), flush=True)
    finally:
        env.close()
    return True


def fork_identity(manifest, parent, mode):
    return {'campaign': digest(manifest), 'parent': parent['checkpoint'],
            'criterion': parent['criterion'], 'seed': parent['seed'], 'mode': mode,
            'parent_counter': parent['counter']}


def save_fork(folder, identity, session, previous, *, schema='rrtl_revision_fork_v1'):
    folder = Path(folder)
    before = identity['parent_counter'] if previous is None else previous['counter']
    rows = session.pending_rows
    if session.interactions-before != len(rows):
        raise ValueError('Fork trace frontier mismatch')
    heads = [] if previous is None else previous['traces']
    if rows:
        raw = gzip.compress(b'\n'.join(json.dumps(row, sort_keys=True, allow_nan=False).encode() for row in rows)+b'\n', mtime=0)
        sha = bytes_hash(raw); path = folder/'traces'/f'{session.interactions}-{sha}.jsonl.gz'
        if not path.exists(): atomic_bytes(path, raw, replace=False)
        heads = [*heads, {'path': str(path.relative_to(folder)), 'sha256': sha,
                         'start': before+1, 'end': session.interactions}]
    audit = session.agent.audit_report(reset=True)
    events = session.agent.events; session.agent.events = []
    session.pending_rows = []
    point = session.interactions-identity['parent_counter']
    checked_write(folder/'audits'/f'{point}.json', {'identity': identity, 'counter': session.interactions,
        'signals': audit, 'events': events, 'metrics': session.agent.revision_metrics(), 'trees': session.agent.tree_export()})
    session.agent.validate_revision()
    payload = {'schema': schema, 'identity': identity, 'session': session.state_dict(),
               'traces': heads, 'counter': session.interactions}
    ref = blob_write(folder, f'checkpoint_{point}', payload)
    checkpoints = {} if previous is None else previous['checkpoints']
    checkpoints = {**checkpoints, str(point): ref}
    status = {'identity': identity, 'counter': session.interactions, 'continuation_decisions': point,
        'checkpoint': ref, 'checkpoints': checkpoints, 'traces': heads,
        'metrics': session.agent.revision_metrics(), 'state': 'partial'}
    checked_write(folder/'status.json', status)
    return status


def restore_fork(folder, status, identity, environment, *, agent_class=RevisionAgent, schema='rrtl_revision_fork_v1'):
    payload = blob_read(folder, status['checkpoint'])
    if payload['schema'] != schema or payload['identity'] != identity or status['identity'] != identity:
        raise ValueError('Fork identity mismatch')
    if payload['counter'] != status['counter'] or payload['traces'] != status['traces']:
        raise ValueError('Fork checkpoint/frontier mismatch')
    expected = identity['parent_counter']+1
    for ref in status['traces']:
        raw = (Path(folder)/ref['path']).read_bytes()
        if bytes_hash(raw) != ref['sha256'] or ref['start'] != expected:
            raise ValueError('Fork trace chain mismatch')
        rows = [json.loads(line) for line in gzip.decompress(raw).splitlines()]
        if len(rows) != ref['end']-ref['start']+1 or any(row['interaction'] != i for row,i in zip(rows,range(ref['start'],ref['end']+1))):
            raise ValueError('Fork trace counter mismatch')
        expected = ref['end']+1
    if expected != status['counter']+1:
        raise ValueError('Fork trace frontier incomplete')
    saved = payload['session']
    if saved['runtime_environment'] != environment.runtime_manifest() or saved['agent_type'] != agent_class.__name__:
        raise ValueError('Fork environment/type mismatch')
    session = OnlineSession.__new__(OnlineSession)
    vars(session).update(saved['collector'])
    session.environment, session.pending_rows = environment, []
    if not isinstance(session.agent, agent_class) or session.agent.interactions != status['counter']:
        raise ValueError('Fork learner counter mismatch')
    session.agent.validate_revision()
    environment.restore(saved['environment'])
    return session


def evaluate_fork_points(output, folder, manifest, bank, status, environment, stop):
    config = manifest['config']; identity = status['identity']
    for point in range(0, status['continuation_decisions']+1, config['evaluation_interval']):
        if stop.requested: return False
        ref = status['checkpoints'][str(point)]
        payload = blob_read(folder, ref)
        agent = payload['session']['collector']['agent']
        for start in bank['starts']:
            if stop.requested: return False
            ei = {**identity, 'checkpoint': ref, 'point': point, 'start_bank': manifest['start_bank'],
                'start': start['id'], 'epsilon': 0., 'horizon': config['evaluation_horizon'],
                'evaluation_seed': config['evaluation_seed']+start['id']}
            evaluation_unit(Path(folder)/'evaluations'/str(point)/str(start['id']), ei, environment,
                agent, start, 0., config['evaluation_horizon'], config['evaluation_seed']+start['id'])
    return True


def continue_one(output, manifest, bank, parent, mode, stop):
    config = manifest['config']; identity = fork_identity(manifest, parent, mode)
    validation_bank = blob_read(output, manifest['validation_bank'])
    folder = Path(output)/'continuations'/parent['criterion']/str(parent['seed'])/mode
    with unit_lock(folder):
        env = default_factory(parent['cell'])(); probe_env = default_factory(parent['cell'])()
        try:
            if (folder/'status.json').exists():
                status = checked_read(folder/'status.json')
                session = restore_fork(folder, status, identity, env)
            else:
                payload = parent_payload(parent)
                session = OnlineSession.restore(payload['session'], parent['cell'], env)
                revision = RevisionConfig(**{**config['revision'], 'mode': mode,
                    'seed': config['revision']['seed']+parent['seed']})
                session.agent = RevisionAgent.fork(session.agent, revision)
                status = save_fork(folder, identity, session, None)
            if not evaluate_fork_points(output, folder, manifest, bank, status, probe_env, stop): return False
            goal = parent['counter']+config['continuation_decisions']
            while session.interactions < goal:
                if stop.requested:
                    if session.pending_rows: save_fork(folder, identity, session, status)
                    return False
                row = session.step()
                row['diagnostic'] = copy.deepcopy(session.agent.last_detail)
                row['objects_after'] = copy.deepcopy(env.last_auxiliary)
                point = session.interactions-parent['counter']
                if point % config['evaluation_interval'] == 0:
                    if mode not in ('normal','frozen'):
                        validation = validation_probes(probe_env, validation_bank,
                            seed=config['probe_seed']+point, blocks=config['probe_blocks'], horizon=config['probe_horizon'])
                        holdout_ref = blob_write(Path(output)/'holdouts', f'probe_{point}', validation)
                        reference_ref = blob_write(folder, f'review_reference_{point}',
                            {'counter': session.interactions, 'roots': copy.deepcopy(session.agent.root_nodes)})
                        reviews = session.agent.maintain(validation)
                        checked_write(folder/'reviews'/f'{point}.json', {'identity': identity,
                            'validation_samples': len(validation), 'validation_seed': config['probe_seed']+point,
                            'holdout': holdout_ref, 'reference': reference_ref,
                            'reviews': reviews, 'validation_used_for_td': False})
                    status = save_fork(folder, identity, session, status)
                    if not evaluate_fork_points(output, folder, manifest, bank, status, probe_env, stop): return False
                    print(json.dumps({'continued': [parent['criterion'],parent['seed'],mode,point],
                                      'revision': session.agent.revision_metrics()}), flush=True)
                elif point % config['save_interval'] == 0:
                    status = save_fork(folder, identity, session, status)
            status = {**status, 'state': 'completed'}
            checked_write(folder/'status.json', status)
            return True
        finally:
            env.close(); probe_env.close()


def continue_group(output, index, stop):
    manifest, bank = load_campaign(output); group = manifest['groups'][index]
    parent = next(p for p in manifest['parents'] if p['criterion'] == group['criterion'] and
                  p['seed'] == group['seed'] and p['counter'] == manifest['config']['parent_counter'])
    for mode in manifest['config']['modes']:
        if not continue_one(output, manifest, bank, parent, mode, stop):
            return False
    return True


def report(output):
    """Completion audit and compact machine-readable observations; no training."""
    output = Path(output); manifest, bank = load_campaign(output); config = manifest['config']
    validation_bank = blob_read(output, manifest['validation_bank'])
    if {s['ram_sha256'] for s in bank['starts']} & {s['ram_sha256'] for s in validation_bank['starts']}:
        raise ValueError('Evaluation/validation RAM starts overlap')
    frozen, continuations, missing = [], [], []
    for parent in manifest['parents']:
        for epsilon in config['evaluation_epsilons']:
            for start in bank['starts']:
                path = output/'frozen'/parent['criterion']/str(parent['seed'])/str(parent['counter'])/str(epsilon)/str(start['id'])/'state.json'
                if not path.exists(): missing.append(str(path.relative_to(output))); continue
                expected = {'campaign': digest(manifest), 'parent': parent['checkpoint'],
                    'start_bank': manifest['start_bank'], 'start': start['id'], 'epsilon': epsilon,
                    'evaluation_seed': config['evaluation_seed']+start['id'],
                    'horizon': config['evaluation_horizon'], 'policy': 'frozen_epsilon_greedy_uniform_ties'}
                unit = verified_evaluation(path, expected)
                if unit['status'] != 'completed': missing.append(str(path.relative_to(output))); continue
                frozen.append({'criterion': parent['criterion'], 'seed': parent['seed'], 'counter': parent['counter'],
                               **{k:v for k,v in unit['result'].items() if k != 'seed'},
                               'evaluation_seed': unit['result']['seed']})
    for group in manifest['groups']:
        for mode in config['modes']:
            folder = output/'continuations'/group['criterion']/str(group['seed'])/mode
            if not (folder/'status.json').exists(): missing.append(str(folder.relative_to(output))); continue
            status = checked_read(folder/'status.json')
            parent = next(p for p in manifest['parents'] if p['criterion'] == group['criterion'] and
                          p['seed'] == group['seed'] and p['counter'] == config['parent_counter'])
            identity = fork_identity(manifest, parent, mode)
            env = default_factory(parent['cell'])()
            try:
                restored = restore_fork(folder, status, identity, env)
                if asdict(restored.agent.revision) != {**config['revision'], 'mode': mode,
                        'seed': config['revision']['seed']+parent['seed']}:
                    raise ValueError('Fork revision configuration changed')
            finally:
                env.close()
            if status['state'] != 'completed' or status['continuation_decisions'] != config['continuation_decisions']:
                missing.append(str(folder.relative_to(output)))
            evaluations, events, signals = [], [], []
            for point in range(0, config['continuation_decisions']+1, config['evaluation_interval']):
                for start in bank['starts']:
                    path = folder/'evaluations'/str(point)/str(start['id'])/'state.json'
                    if not path.exists(): missing.append(str(path.relative_to(output))); continue
                    expected = {**identity, 'checkpoint': status['checkpoints'][str(point)],
                        'point': point, 'start_bank': manifest['start_bank'], 'start': start['id'],
                        'epsilon': 0., 'horizon': config['evaluation_horizon'],
                        'evaluation_seed': config['evaluation_seed']+start['id']}
                    saved = verified_evaluation(path, expected)
                    if saved['status'] != 'completed': missing.append(str(path.relative_to(output))); continue
                    evaluations.append({'point': point, **saved['result']})
            for path in sorted((folder/'audits').glob('*.json')):
                if path.stem not in status['checkpoints']:
                    continue  # Uncommitted orphan audit from an interrupted write.
                saved = checked_read(path); events.extend(saved['events'])
                for signal in ('updated_q','target','td_error'):
                    rows = [r['signals'][signal] for r in saved['signals']]
                    valid = [r for r in rows if r['n'] > 128 and r['p_anova'] is not None]
                    signals.append({'point': int(path.stem), 'signal': signal, 'candidate_windows': len(rows),
                        'eligible_sample_windows': len(valid),
                        'anova_below_threshold': sum(r['p_anova'] <= .0001 for r in valid),
                        'historical_below_threshold': sum(r['p_historical'] is not None and r['p_historical'] <= .0001 for r in valid),
                        'median_effect': float(np.median([r['effect_fraction'] for r in valid])) if valid else None})
            if sum(e['kind']=='grow' for e in events) != status['metrics']['grows_total']:
                raise ValueError('Missing growth events')
            for kind, counter in [('prune_review','prunes_total'),('replacement_review','replacements_total')]:
                if sum(e['kind']==kind and e['gate']['accepted'] for e in events) != status['metrics'][counter]:
                    raise ValueError('Revision event/counter mismatch')
            for path in (folder/'reviews').glob('*.json'):
                if path.stem not in status['checkpoints']: continue
                review = checked_read(path)
                if review['validation_used_for_td'] or review['identity'] != identity:
                    raise ValueError('Invalid holdout provenance')
                if len(blob_read(output/'holdouts',review['holdout'])) != review['validation_samples']:
                    raise ValueError('Holdout sample mismatch')
                if blob_read(folder,review['reference'])['counter'] != config['parent_counter']+int(path.stem):
                    raise ValueError('Review reference counter mismatch')
            continuations.append({**group, 'mode': mode, 'metrics': status['metrics'],
                'counter': status['counter'], 'state': status['state'], 'evaluations': evaluations,
                'events': events, 'signal_audit': signals})
    result = {'schema': 'rrtl_revision_report_v1', 'campaign': digest(manifest),
        'complete': not missing, 'missing': missing, 'start_diversity': manifest['start_diversity'],
        'validation_start_diversity': {k: validation_bank[k] for k in ('unique_ram','unique_encoded')},
        'evaluation_validation_start_ram_disjoint': True,
        'config': config, 'frozen': frozen, 'continuations': continuations,
        'limitations': ['Diagnostic snapshot continuations, not complete-game benchmark scores.',
            'Three selected seed case studies do not estimate population superiority.',
            'Block-bootstrap gates are exploratory and repeated selection is not calibrated.',
            'Bellman loss improvements do not guarantee return improvement.', 'No reserved test executed.']}
    checked_write(output/'report.json', result)
    print(json.dumps({'complete': not missing, 'missing_units': len(missing),
                      'frozen_episodes': len(frozen), 'continuations': len(continuations)}))
    return not missing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('prepare','frozen','continue','report'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--historical', type=Path)
    parser.add_argument('--anova', type=Path)
    parser.add_argument('--group', type=int, default=0)
    args = parser.parse_args()
    with StopRequest() as stop:
        if args.stage == 'prepare':
            prepare(args.config, args.historical, args.anova, args.output); complete = True
        elif args.stage == 'frozen': complete = frozen_group(args.output, args.group, stop)
        elif args.stage == 'continue': complete = continue_group(args.output, args.group, stop)
        else: complete = report(args.output)
    raise SystemExit(0 if complete else 3)


if __name__ == '__main__': main()
