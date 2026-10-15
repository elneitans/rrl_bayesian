"""B12 one-cell training and independent checkpoint evaluation. No reserved-test route."""
import argparse
import copy
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import resource
import signal
import socket
import sys
import time
import uuid

import numpy as np

from bayesian_rrtl.comparison import evaluation_summary
from bayesian_rrtl.online.learners import LEARNERS
from bayesian_rrtl.online.protocol import (authorize, authorized_stop, dry_run, freeze, load,
                                         runtime, select_cell, source_hashes)
from bayesian_rrtl.online.session import OnlineSession
from bayesian_rrtl.online.multigame_protocol import cell_protocol
from bayesian_rrtl.online.storage import (binding, checkpoint_metadata, cleanup, committed_checkpoints,
    inventory, load_checkpoint, load_frontier, publish, stream_rows)
from experiments.pong_analysis_runtime import (Budget, checked_read, checked_write,
                                               digest, episode, unit_lock)
from experiments.pong_common_contract import CommonPongConfig, CommonPongFactory


class StopRequest:
    def __init__(self):
        self.requested = False
        self.signal = None
        self.old = {}

    def handler(self, signum, frame):
        self.requested = True
        self.signal = signum

    def __enter__(self):
        for signum in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT):
            self.old[signum] = signal.signal(signum, self.handler)
        return self

    def __exit__(self, *args):
        for signum, handler in self.old.items():
            signal.signal(signum, handler)


def peak_rss():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)


def attempt_measurements(paths):
    """Read attempts without inventing final measurements for killed/live workers.

    A missing final record cannot distinguish SIGKILL from a running process.
    Preserve the original file and expose its unknown fields in the report.
    """
    records = []
    for path in sorted(paths):
        saved = checked_read(path)
        finalized = saved.get('ended_unix') is not None
        records.append({**saved, 'attempt_file': path.name,
                        'record_status': 'finalized' if finalized else 'incomplete',
                        'ended_unix': saved.get('ended_unix'),
                        'seconds': saved.get('seconds'),
                        'peak_rss_bytes': saved.get('peak_rss_bytes')})
    peaks = [r['peak_rss_bytes'] for r in records if r['peak_rss_bytes'] is not None]
    known_seconds = [r['seconds'] for r in records if r['seconds'] is not None]
    peak_complete = bool(records) and all(
        r['record_status'] == 'finalized' and r['peak_rss_bytes'] is not None for r in records)
    seconds_complete = bool(records) and all(
        r['record_status'] == 'finalized' and r['seconds'] is not None for r in records)
    return {'records': records,
            'incomplete_attempts': sum(r['record_status'] == 'incomplete' for r in records),
            'peak_rss_bytes': max(peaks) if peak_complete else None,
            'peak_rss_bytes_known_max': max(peaks) if peaks else None,
            'seconds_known_sum': sum(known_seconds),
            'seconds_complete': seconds_complete}


def default_factory(cell):
    if 'game' in cell:
        from bayesian_rrtl.online.environments import CommonGameFactory, GameEnvironmentConfig
        return CommonGameFactory(GameEnvironmentConfig(**cell['environment']))
    return CommonPongFactory(CommonPongConfig(**cell['environment']))


def required_points(manifest, counter=None, cell=None):
    p = cell_protocol(manifest, cell) if cell is not None else manifest['protocol']
    return sorted({point for name, g in p['evaluation'].items() if name != 'reserved_test'
                   for point in g['checkpoints'] if counter is None or point <= counter})


def validate_status(status, manifest, cell):
    if status['binding'] != binding(manifest, cell):
        raise ValueError('Cell status identity mismatch')
    return status



def validate_extension(seconds, extension_id, reason):
    if (type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0
            or bool(seconds) != bool(extension_id)):
        raise ValueError('Time extension requires positive finite seconds and an extension ID')
    if seconds and (not isinstance(extension_id, str) or not extension_id.strip()
                    or not isinstance(reason, str) or not reason.strip()):
        raise ValueError('Budget extension needs an explicit recorded ID and reason')


def validate_attempt_seconds(seconds):
    if seconds is not None and (type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds <= 0):
        raise ValueError('Attempt seconds must be finite and positive')

def train_cell(output, selection, *, factory=None, extend_seconds=0., extension_id=None,
               extension_reason=None, decision_limit=None, stop=None, after_publish=None, attempt_seconds=None):
    output = Path(output)
    manifest = load(output)
    cell = select_cell(manifest, selection)  # index checked before creating output or lock
    limit = authorized_stop(output, manifest, cell)
    validate_extension(extend_seconds, extension_id, extension_reason)
    validate_attempt_seconds(attempt_seconds)
    p = cell_protocol(manifest, cell)
    folder = output/cell['id']
    with unit_lock(folder):
        status = validate_status(checked_read(folder/'status.json'), manifest, cell) if (folder/'status.json').exists() else None
        if status is not None and status['counter'] == p['max_decisions'] and status['state'] == 'completed':
            load_frontier(folder, status, binding(manifest, cell))
            return status
        if extend_seconds:
            path = folder/'budget_reviews.json'
            reviews = checked_read(path) if path.exists() else []
            record = {'id': extension_id, 'seconds': extend_seconds, 'reason': extension_reason}
            previous = next((r for r in reviews if r['id'] == extension_id), None)
            if previous and previous != record:
                raise ValueError('Changed extension review')
            if not previous:
                checked_write(path, reviews+[record])
        budget = Budget(folder, p['budgets']['training_seconds_per_cell'],
                        extend_seconds=extend_seconds, extension_id=extension_id)
        attempt_path = folder/'attempts'/(uuid.uuid4().hex+'.json')
        attempt = {'started_unix': time.time(), 'node': socket.gethostname(), 'runtime': runtime(cell['learner']),
                   'resources': {key: os.environ.get(key) for key in ('SLURM_JOB_ID', 'SLURM_ARRAY_TASK_ID',
                       'SLURM_CPUS_PER_TASK', 'SLURM_MEM_PER_NODE', 'SLURM_JOB_PARTITION', 'SLURM_JOB_QOS')},
                   'environment': {key: os.environ.get(key) for key in ('CONDA_DEFAULT_ENV', 'OMP_NUM_THREADS',
                        'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')}, 'executable': sys.executable,
                   'resume_counter': status['counter'] if status else 0, 'budget_extension': extension_id,
                   'attempt_seconds': attempt_seconds, 'state': 'running',
                   'ended_unix': None, 'seconds': None, 'peak_rss_bytes': None}
        checked_write(attempt_path, attempt)
        env = session = None
        started = time.perf_counter()
        invocation_steps = 0
        checkpoints = set(required_points(manifest, cell=cell)) | set(p['milestones'])
        try:
            env = (factory or default_factory(cell))()
            if status and status.get('checkpoint'):
                payload, _ = load_frontier(folder, status, binding(manifest, cell))
                session = OnlineSession.restore(payload['session'], cell, env)
                session.phase_seconds['checkpoint_logs'] = status['metrics']['phase_seconds']['checkpoint_logs']
            else:
                session = OnlineSession(cell, env, cache_bytes=p['cache_bytes'])
                status = publish(folder, manifest, cell, session, None, state='partial', reason='initial')
                if after_publish:
                    after_publish(session.interactions)
            while session.interactions < limit:
                attempt_remaining = (float('inf') if attempt_seconds is None else
                                     max(0., attempt_seconds-(time.perf_counter()-started)))
                remaining = min(budget.remaining, attempt_remaining)
                reason = None
                if stop is not None and stop.requested:
                    reason = 'interrupted_signal'
                elif decision_limit is not None and invocation_steps >= decision_limit:
                    reason = 'administrative_pause'
                elif budget.remaining <= 0:
                    reason = 'training_budget_exhausted'
                elif attempt_remaining <= 0:
                    reason = 'attempt_time_exhausted'
                elif session.next_fit_due() and remaining < session.last_fit_cost+p['budgets']['fit_margin_seconds']:
                    reason = 'insufficient_time_before_fit'
                if reason:
                    status = publish(folder, manifest, cell, session, status, state='partial', reason=reason)
                    return status
                session.step()
                invocation_steps += 1
                if session.interactions % p['checkpoint_interval'] == 0 or session.interactions in checkpoints:
                    status = publish(folder, manifest, cell, session, status, state='partial', reason='in_progress')
                    budget.charge()
                    if after_publish:
                        after_publish(session.interactions)
            status = publish(folder, manifest, cell, session, status,
                state='completed' if limit == p['max_decisions'] else 'partial',
                reason=None if limit == p['max_decisions'] else 'authorized_milestone_reached')
            return status
        except Exception as exc:
            if status:
                # Preserve last committed model/log frontier; never save an incomplete update.
                status = {**status, 'state': 'failed', 'reason': f'{type(exc).__name__}: {exc}'}
                checked_write(folder/'status.json', status)
            else:
                status = {'binding': binding(manifest, cell), 'counter': 0, 'checkpoint': None,
                          'log_head': None, 'state': 'failed', 'reason': f'{type(exc).__name__}: {exc}'}
                checked_write(folder/'status.json', status)
            raise
        finally:
            budget.charge()
            if env:
                env.close()
            attempt.update(seconds=time.perf_counter()-started, ended_unix=time.time(),
                           peak_rss_bytes=peak_rss(), confirmed_counter=status['counter'] if status else None,
                           state=status['state'] if status else 'failed', reason=status.get('reason') if status else 'initialization_failure')
            checked_write(attempt_path, attempt)


class GreedyController:
    def __init__(self, agent, learner):
        self.agent = copy.deepcopy(agent)
        self.learner = learner
        if LEARNERS[learner].batch:
            self.agent.cache.clear()

    def action(self, state, env, rng):
        if LEARNERS[self.learner].batch:
            q = self.agent.predict_q(self.agent.encode_state(state))[0]
        else:
            q = self.agent.predict_q(state)
        return int(rng.choice(np.flatnonzero(q == q.max())))


def checkpoint_at(folder, status, point):
    for ref in committed_checkpoints(folder, status):
        meta = checkpoint_metadata(folder, ref)
        if meta['counter'] == point:
            return ref
    raise ValueError(f'No committed checkpoint at {point}')


def evaluation_identity(manifest, cell, ref, group, seed):
    g = cell_protocol(manifest, cell)['evaluation'][group]
    return {'schema': 'pong_checkpoint_evaluation_v2', 'campaign': digest(manifest), 'cell': cell['hash'],
            'checkpoint': ref, 'group': group, 'seeds': g['seeds'], 'seed': seed, 'horizon': g['horizon'],
            'policy': 'greedy_uniform_ties_v1', 'runtime': runtime(cell['learner'])}



@contextmanager
def evaluation_attempt(folder, identity):
    folder = Path(folder)
    with unit_lock(folder/'attempts'):
        state = checked_read(folder/'state.json') if (folder/'state.json').exists() else None
        if state and state['status'] == 'completed':
            yield
            return
        path = folder/'attempts'/(uuid.uuid4().hex+'.json')
        start = time.perf_counter()
        record = {'started_unix': time.time(), 'state': 'running', 'identity': digest(identity),
                  'node': socket.gethostname(), 'runtime': identity['runtime'], 'executable': sys.executable,
                  'resources': {k: os.environ.get(k) for k in ('SLURM_JOB_ID', 'SLURM_CPUS_PER_TASK',
                       'SLURM_MEM_PER_NODE', 'SLURM_JOB_PARTITION', 'SLURM_JOB_QOS')},
                  'environment': {k: os.environ.get(k) for k in ('CONDA_DEFAULT_ENV', 'OMP_NUM_THREADS',
                       'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')},
                  'resume_steps': state['result']['steps'] if state else 0}
        checked_write(path, record)
        try:
            yield
        except Exception as exc:
            record.update(state='failed', reason=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            saved = checked_read(folder/'state.json') if (folder/'state.json').exists() else None
            if record['state'] != 'failed':
                record.update(state=saved['status'] if saved else 'partial',
                              reason=saved['reason'] if saved else 'no_checkpoint')
            record.update(seconds=time.perf_counter()-start, ended_unix=time.time(), peak_rss_bytes=peak_rss())
            checked_write(path, record)

def _evaluate_cell(output, selection, *, point=None, group=None, factory=None,
                  extend_seconds=0., extension_id=None, extension_reason=None, stop=None):
    output = Path(output)
    manifest = load(output)
    cell = select_cell(manifest, selection)
    if group not in (None, 'intermediate', 'final_development'):
        raise ValueError('Reserved test is unavailable in block B')
    validate_extension(extend_seconds, extension_id, extension_reason)
    folder = output/cell['id']
    status = validate_status(checked_read(folder/'status.json'), manifest, cell)
    load_frontier(folder, status, binding(manifest, cell))
    p = cell_protocol(manifest, cell)
    results = []
    groups = [group] if group else ['intermediate', 'final_development']
    for name in groups:
        spec = p['evaluation'][name]
        points = [point] if point is not None else [n for n in spec['checkpoints'] if n <= status['counter']]
        for n in points:
            if n not in spec['checkpoints']:
                raise ValueError('Checkpoint is not in predeclared evaluation calendar')
            ref = checkpoint_at(folder, status, n)
            payload, _ = load_checkpoint(folder, ref, binding(manifest, cell))
            agent = payload['session']['collector']['agent']
            eval_factory = (factory or default_factory(cell)).with_horizon(spec['horizon'])
            rows = []
            for seed in spec['seeds']:
                identity = evaluation_identity(manifest, cell, ref, name, seed)
                unit_folder = folder/'evaluations'/name/str(n)/str(seed)
                if extend_seconds:
                    with unit_lock(unit_folder):
                        review_path = unit_folder/'budget_review.json'
                        review = {'id': extension_id, 'seconds': extend_seconds, 'reason': extension_reason}
                        # Preserve every review rather than overwriting prior justification.
                        reviews = checked_read(review_path) if review_path.exists() else []
                        old = next((r for r in reviews if r['id'] == extension_id), None)
                        if old and old != review:
                            raise ValueError('Evaluation extension review changed')
                        if not old:
                            checked_write(review_path, reviews+[review])
                with evaluation_attempt(unit_folder, identity):
                    # Atomic per-episode units give partial test-like evaluation resumption without reruns.
                    row = episode(eval_factory, GreedyController(agent, cell['learner']), unit_folder, identity,
                        seed=seed, horizon=spec['horizon'], seconds=p['budgets']['evaluation_seconds_per_episode'],
                        rng_seed=seed, extend_seconds=extend_seconds, extension_id=extension_id, stop=stop,
                        pong_points=cell.get('game','Pong') == 'Pong')
                    if stop is not None and stop.requested:
                        return results+[{'group': name, 'counter': n, 'checkpoint': ref,
                                         'state': 'partial', 'episodes': rows+[row], 'reason': 'interrupted_signal'}]
                rows.append(row)
            results.append({'group': name, 'counter': n, 'checkpoint': ref,
                            'state': 'completed' if all(r['status'] == 'completed' for r in rows) else 'partial',
                            'episodes': rows})
    return results



def evaluate_cell(output, selection, *, point=None, group=None, factory=None,
                  extend_seconds=0., extension_id=None, extension_reason=None, stop=None):
    manifest = load(output)
    cell = select_cell(manifest, selection)
    if group not in (None, 'intermediate', 'final_development'):
        raise ValueError('Reserved test is unavailable in block B')
    folder = Path(output)/cell['id']
    if not (folder/'status.json').exists():
        raise ValueError('Training has not published a checkpoint')
    path = folder/'evaluation_workers'/(uuid.uuid4().hex+'.json')
    started = time.perf_counter()
    record = {'started_unix': time.time(), 'state': 'running', 'campaign': digest(manifest),
              'cell': cell['hash'], 'point': point, 'group': group, 'node': socket.gethostname(),
              'runtime': runtime(cell['learner']), 'executable': sys.executable,
              'resources': {k: os.environ.get(k) for k in ('SLURM_JOB_ID', 'SLURM_CPUS_PER_TASK',
                  'SLURM_MEM_PER_NODE', 'SLURM_JOB_PARTITION', 'SLURM_JOB_QOS')},
              'environment': {k: os.environ.get(k) for k in ('CONDA_DEFAULT_ENV', 'OMP_NUM_THREADS',
                  'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')}}
    checked_write(path, record)
    try:
        result = _evaluate_cell(output, selection, point=point, group=group, factory=factory,
            extend_seconds=extend_seconds, extension_id=extension_id, extension_reason=extension_reason, stop=stop)
        record['state'] = 'completed' if result and all(r['state'] == 'completed' for r in result) else 'partial'
        return result
    except Exception as exc:
        record.update(state='failed', reason=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        record.update(seconds=time.perf_counter()-started, ended_unix=time.time(), peak_rss_bytes=peak_rss())
        checked_write(path, record)

def consolidate(output):
    output = Path(output)
    manifest = load(output)
    rows = []
    training_seconds = evaluation_seconds = evaluation_budget_seconds = 0.
    for cell in manifest['cells']:
        p = cell_protocol(manifest, cell)
        folder = output/cell['id']
        if not (folder/'status.json').exists():
            rows.append({'cell': cell['id'], 'training_state': 'pending', 'comparison_state': 'pending',
                         'evaluations': [], 'pending_evaluations': []})
            continue
        status = validate_status(checked_read(folder/'status.json'), manifest, cell)
        if not status.get('checkpoint'):
            rows.append({'cell': cell['id'], 'training_state': status['state'],
                         'comparison_state': 'failed', 'reason': status['reason'], 'counter': 0,
                         'evaluations': [], 'pending_evaluations': []})
            continue
        load_frontier(folder, status, binding(manifest, cell))
        inv = inventory(folder, status, required_points(manifest, cell=cell))
        pending, evaluations = [], []
        for group, g in p['evaluation'].items():
            if group == 'reserved_test':
                continue
            for n in g['checkpoints']:
                if n > status['counter']:
                    pending.append({'group': group, 'counter': n, 'reason': 'checkpoint_pending'})
                    continue
                ref = checkpoint_at(folder, status, n)
                episodes = []
                for seed in g['seeds']:
                    unit_folder = folder/'evaluations'/group/str(n)/str(seed)
                    identity_path = unit_folder/'identity.json'
                    if not identity_path.exists():
                        pending.append({'group': group, 'counter': n, 'seed': seed, 'reason': 'evaluation_pending'})
                        continue
                    from experiments.pong_analysis_runtime import WorkUnit
                    # Read-only validation: no creation or evaluation in report path.
                    if checked_read(identity_path) != evaluation_identity(manifest, cell, ref, group, seed):
                        raise ValueError('Evaluation identity mismatch')
                    try:
                        with unit_lock(unit_folder):
                            unit = WorkUnit(unit_folder, checked_read(identity_path))
                            saved = unit.load()
                    except RuntimeError as exc:
                        if 'Concurrent writer' not in str(exc):
                            raise
                        pending.append({'group': group, 'counter': n, 'seed': seed,
                                        'reason': 'evaluation_running'})
                        continue
                    if saved is None or saved['status'] != 'completed':
                        attempts = [checked_read(path) for path in (unit_folder/'attempts').glob('*.json')]
                        failed = bool(attempts and max(attempts, key=lambda a: a['started_unix']).get('state') == 'failed')
                        pending.append({'group': group, 'counter': n, 'seed': seed,
                                        'reason': 'evaluation_failed' if failed else 'evaluation_partial'})
                        continue
                    ep = saved['result']
                    episodes.append({**ep, 'return': ep['raw_return']})
                if len(episodes) == len(g['seeds']):
                    statistics = evaluation_summary(episodes, g['horizon'])
                    seconds = sum(r['seconds'] for r in episodes)
                    evaluations.append({'group': group, 'counter': n, 'checkpoint': ref,
                                        'training_active_seconds': checkpoint_metadata(folder, ref).get('training_active_seconds'),
                                        'seconds': seconds, 'report': statistics,
                                        'floor_result': (all(r['terminated'] and r['raw_return'] <= -21 for r in episodes)
                                                         if cell.get('game','Pong') == 'Pong' else None)})
        unit_evaluation_seconds = sum(checked_read(b)['elapsed_seconds']
            for b in (folder/'evaluations').rglob('budget.json'))
        workers = attempt_measurements((folder/'evaluation_workers').glob('*.json'))
        worker_seconds = workers['seconds_known_sum']
        evaluation_seconds += worker_seconds
        evaluation_budget_seconds += unit_evaluation_seconds
        training_attempts = attempt_measurements((folder/'attempts').glob('*.json'))
        elapsed = checked_read(folder/'budget.json')['elapsed_seconds']
        training_seconds += elapsed
        reconstructed = {'decisions': 0, 'ended_episodes': 0, 'raw_return': 0., 'points_won': 0., 'points_lost': 0.}
        for row in stream_rows(folder, status['log_head']):
            reconstructed['decisions'] += 1
            reconstructed['ended_episodes'] += int(row['episode_end'])
            reconstructed['raw_return'] += row['raw_reward']
            reconstructed['points_won'] += max(row['raw_reward'], 0.)
            reconstructed['points_lost'] += max(-row['raw_reward'], 0.)
        if cell.get('game','Pong') != 'Pong':
            reconstructed.pop('points_won',None)
            reconstructed.pop('points_lost',None)
        rows.append({'cell': cell['id'], 'game': cell.get('game','Pong'),
                     'training_state': status['state'], 'reason': status['reason'],
                     'comparison_state': ('failed' if status['state'] == 'failed' or any(v['reason'] == 'evaluation_failed' for v in pending) else
                         'completed' if status['state'] == 'completed' and not pending else
                         'pending_evaluation' if status['state'] == 'completed' else 'partial'),
                     'counter': status['counter'], 'metrics': status['metrics'], 'logs': reconstructed,
                     'training_attempts': len(training_attempts['records']),
                     'training_attempt_records': training_attempts['records'],
                     'unfinished_training_attempts': training_attempts['incomplete_attempts'],
                     'training_seconds_complete': training_attempts['seconds_complete'],
                     'training_attempt_seconds_known_sum': training_attempts['seconds_known_sum'],
                     'peak_rss_bytes': training_attempts['peak_rss_bytes'],
                     'peak_rss_bytes_known_max': training_attempts['peak_rss_bytes_known_max'],
                     'training_seconds': elapsed, 'evaluation_seconds': worker_seconds, 'evaluation_budget_seconds': unit_evaluation_seconds,
                     'evaluation_seconds_complete': workers['seconds_complete'],
                     'unfinished_evaluation_attempts': workers['incomplete_attempts'], 'inventory': inv, 'evaluations': evaluations,
                     'pending_evaluations': pending})
    report = {'schema': 'pong_progressive_report_v2', 'campaign': digest(manifest), 'cells': rows,
              'completed': all(r['comparison_state'] == 'completed' for r in rows),
              'accounting': {'training_seconds_per_unit_sum': training_seconds,
                             'training_seconds_complete': all(r.get('training_seconds_complete', False) for r in rows),
                             'evaluation_worker_seconds_known_sum': evaluation_seconds,
                             'evaluation_worker_seconds_complete': all(r.get('evaluation_seconds_complete', False) for r in rows),
                             'evaluation_budget_seconds_per_episode_sum': evaluation_budget_seconds},
              'limitations': ['A floor result does not establish equivalence.',
                  'Smoke is technical only; primary result is frozen final checkpoint.',
                  'Operational sums are resource consumption, not concurrent wall time.',
                  'Unfinalized attempts have unknown end/duration/RSS; budget time is a recorded lower bound after abrupt death.'],
              'reserved_test_executed': (output/'final_test'/'usage.json').exists(),
              'reserved_test_executed_by_report': False}
    if manifest['schema'] == 'atari_comparison_v3':
        from bayesian_rrtl.evaluation import comparative_report
        report['analysis'] = comparative_report(manifest, rows)
    with unit_lock(output/'reports'):
        checked_write(output/'reports'/'report.json', report)
    return report


def measurements(output):
    manifest = load(output)
    report = consolidate(output)
    by_id = {r['cell']: r for r in report['cells']}
    rows = []
    for cell in manifest['cells']:
        p = cell_protocol(manifest, cell)
        r = by_id[cell['id']]
        if not r.get('counter'):
            continue
        folder = Path(output)/cell['id']
        status = checked_read(folder/'status.json')
        payload, _ = load_frontier(folder, status, binding(manifest, cell))
        rows.append({'environment': cell['environment'], 'learner': cell['learner'],
                     'learning_without_seed': {k: v for k, v in cell['learning'].items() if k != 'seed'},
                     'runtime_environment': payload['session']['runtime_environment'],
                     'catalog_hash': payload['session']['collector']['agent'].encoder.catalog_hash,
                     'cache_bytes': p['cache_bytes'],
                     'evaluation_horizons': {k:g['horizon'] for k,g in p['evaluation'].items()
                                             if k != 'reserved_test'},
                     'estimator': cell['estimator'], 'runtime': runtime(cell['learner']),
                     'source_hashes': source_hashes(), 'decisions': r['counter'],
                     'training_seconds': r['training_seconds'] if r['training_seconds_complete'] else None,
                     'training_seconds_recorded_lower_bound': r['training_seconds'],
                     'training_attempts': r['training_attempt_records'],
                     'unfinished_training_attempts': r['unfinished_training_attempts'],
                     'log_bytes': r['inventory']['log_bytes'],
                     'checkpoint_bytes': r['inventory']['checkpoint_bytes'],
                     'checkpoint_blob_bytes_range': [min(a['bytes'] for a in r['inventory']['artifacts']),
                                                     max(a['bytes'] for a in r['inventory']['artifacts'])],
                     'evaluation_episode_seconds': [ep['seconds'] for e in r['evaluations'] for ep in e['report']['episodes']],
                     'evaluation_worker_seconds_per_completed_episode': r['evaluation_seconds']/sum(
                         len(e['report']['episodes']) for e in r['evaluations']) if r['evaluations'] and r['evaluation_seconds_complete'] else None,
                     'peak_rss_bytes': r['peak_rss_bytes'],
                     'peak_rss_bytes_known_max': r['peak_rss_bytes_known_max'],
                     'evaluation_seconds': r['evaluation_seconds'] if r['evaluation_seconds_complete'] else None,
                     'evaluation_seconds_known_sum': r['evaluation_seconds'],
                     'unfinished_evaluation_attempts': r['unfinished_evaluation_attempts']})
    return rows



def cost_reference(path):
    if path is None:
        return None
    path=Path(path)
    if path.is_file():
        records=json.loads(path.read_text())
        if not isinstance(records,list):
            raise ValueError('Cost measurement export must contain a list of measurements')
        return records
    return measurements(path)

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('dry-run', 'freeze', 'authorize', 'train-cell', 'resume',
                                          'evaluate', 'report', 'inventory', 'cleanup', 'measurements', 'prepare-test', 'authorize-test', 'test', 'report-test'), required=True)
    parser.add_argument('--protocol', type=Path)
    parser.add_argument('--campaign', type=Path)
    parser.add_argument('--cell')
    parser.add_argument('--point', type=int)
    parser.add_argument('--group', choices=('intermediate', 'final_development'))
    parser.add_argument('--stop-at', type=int)
    parser.add_argument('--game', choices=('Pong','Breakout','DemonAttack'))
    parser.add_argument('--reason')
    parser.add_argument('--extend-seconds', type=float, default=0.)
    parser.add_argument('--extension-id')
    parser.add_argument('--attempt-seconds', type=float, help='Operational time remaining for this training attempt; recorded, no counter reset')
    parser.add_argument('--cost-reference', type=Path)
    parser.add_argument('--selection-record', type=Path)
    parser.add_argument('--confirm-test', action='store_true')
    args = parser.parse_args(argv)
    if args.attempt_seconds is not None and args.stage not in ('train-cell', 'resume'):
        parser.error('--attempt-seconds is only for a training attempt')
    if args.stage in ('dry-run', 'freeze'):
        if args.protocol is None:
            parser.error('Protocol required for preparation')
        raw = json.loads(args.protocol.read_text())
        if args.stage == 'dry-run':
            reference = cost_reference(args.cost_reference)
            result = dry_run(raw, reference)
        else:
            if args.campaign is None:
                parser.error('Campaign path required')
            result = freeze(raw, args.campaign)
    else:
        if args.campaign is None or args.protocol:
            parser.error('Use only frozen campaign identity after preparation')
        if args.stage in ('prepare-test','authorize-test','test','report-test'):
            from bayesian_rrtl.online import final_test
            if args.stage == 'prepare-test':
                if not args.selection_record:
                    parser.error('prepare-test requires --selection-record output path')
                result = final_test.prepare(args.campaign,args.selection_record)
            elif args.stage == 'authorize-test':
                if not args.selection_record:
                    parser.error('authorize-test requires --selection-record')
                result = final_test.authorize(args.campaign,args.selection_record,
                    confirmed=args.confirm_test,reason=args.reason)
            elif args.stage == 'report-test':
                result = final_test.report(args.campaign)
            else:
                if args.cell is None:
                    parser.error('Explicit test requires one --cell')
                result = final_test.evaluate(args.campaign,args.cell,confirmed=args.confirm_test,
                    extend_seconds=args.extend_seconds,extension_id=args.extension_id,reason=args.reason)
        elif args.stage == 'authorize':
            if args.stop_at is None or not args.reason:
                parser.error('Authorize requires stop-at and reason')
            result = authorize(args.campaign, args.stop_at, args.reason,
                cost_reference(args.cost_reference), game=args.game)
        elif args.stage in ('train-cell', 'resume'):
            if args.cell is None:
                parser.error('One cell required')
            with StopRequest() as stop:
                result = train_cell(args.campaign, args.cell, stop=stop, extend_seconds=args.extend_seconds,
                    extension_id=args.extension_id, extension_reason=args.reason, attempt_seconds=args.attempt_seconds)
        elif args.stage == 'evaluate':
            if args.cell is None:
                parser.error('One cell required')
            with StopRequest() as stop:
                result = evaluate_cell(args.campaign, args.cell, point=args.point, group=args.group,
                    extend_seconds=args.extend_seconds, extension_id=args.extension_id,
                    extension_reason=args.reason, stop=stop)
        elif args.stage in ('inventory', 'cleanup'):
            m = load(args.campaign)
            cell = select_cell(m, args.cell)
            folder = args.campaign/cell['id']
            # Cell lock excludes training; every allowed evaluation point is always retained.
            with unit_lock(folder):
                s = validate_status(checked_read(folder/'status.json'), m, cell)
                result = (cleanup if args.stage == 'cleanup' else inventory)(folder, s, required_points(m, cell=cell))
        elif args.stage == 'measurements':
            result = measurements(args.campaign)
        else:
            result = consolidate(args.campaign)
    print(json.dumps(result, indent=2, allow_nan=False))
    if args.stage in ('train-cell', 'resume'):
        return 0 if result['state'] == 'completed' else 3
    if args.stage in ('test','report-test'):
        return 0 if result['state'] == 'completed' else 3
    if args.stage == 'evaluate':
        return 0 if result and all(r['state'] == 'completed' for r in result) else 3
    if args.stage == 'report':
        return 0 if result['completed'] else 3
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
