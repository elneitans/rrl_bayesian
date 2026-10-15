"""B11: immutable bounded log segments and checkpoint/segment commit frontier."""
import json
from pathlib import Path
import pickle
import time

from experiments.pong_analysis_runtime import (atomic_bytes, bytes_hash, checked_read,
                                              checked_write, digest)
from bayesian_rrtl.online.protocol import runtime


def verified_bytes(folder, ref):
    path = Path(folder)/ref['path']
    if path.resolve().parent not in ((Path(folder)/'logs').resolve(), (Path(folder)/'checkpoints').resolve()):
        raise ValueError('Invalid artifact path')
    raw = path.read_bytes()
    if bytes_hash(raw) != ref['sha256']:
        raise ValueError('Artifact checksum mismatch')
    return raw


def segment_refs(folder, head):
    refs = []
    last_start = None
    while head is not None:
        raw = verified_bytes(folder, head)
        header = json.loads(raw.split(b'\n', 1)[0])
        if (header['schema'] != 'pong_log_segment_v2' or header['end'] != head['end']
                or (last_start is not None and header['end'] != last_start-1)):
            raise ValueError('Invalid segment frontier')
        refs.append(head)
        last_start = header['start']
        head = header['previous']
    if refs and last_start != 1:
        raise ValueError('Logs do not begin at decision one')
    return list(reversed(refs))


def stream_rows(folder, head):
    """Memory bounded by one segment plus segment references, never whole history."""
    for ref in segment_refs(folder, head):
        lines = verified_bytes(folder, ref).splitlines()
        header = json.loads(lines[0])
        if len(lines)-1 != header['end']-header['start']+1:
            raise ValueError('Segment length mismatch')
        for index, line in enumerate(lines[1:], header['start']):
            row = json.loads(line)
            if row['interaction'] != index or row['transition_id'] != index-1:
                raise ValueError('Duplicated/gapped transition')
            yield row


def binding(manifest, cell):
    return {'campaign': digest(manifest), 'cell': cell['hash'], 'learner': cell['learner'],
            'runtime': runtime(cell['learner'])}


def checkpoint_metadata(folder, ref):
    path = Path(folder)/(ref['path']+'.json')
    meta = checked_read(path)
    if meta['checkpoint'] != ref:
        raise ValueError('Checkpoint metadata/reference mismatch')
    return meta


def load_checkpoint(folder, ref, expected):
    meta = checkpoint_metadata(folder, ref)
    if meta['binding'] != expected:
        raise ValueError('Checkpoint campaign/type/runtime mismatch')
    payload = pickle.loads(verified_bytes(folder, ref))  # trusted local, checksum verified first
    if payload['binding'] != expected or payload['counter'] != meta['counter'] or payload['log_head'] != meta['log_head']:
        raise ValueError('Checkpoint frontier mismatch')
    count = 0
    for row in stream_rows(folder, payload['log_head']):
        count = row['interaction']
    if count != payload['counter'] or payload['session']['collector']['agent'].interactions != count:
        raise ValueError('Model/counter/log frontier mismatch')
    return payload, meta


def load_frontier(folder, status, expected):
    payload, metadata = load_checkpoint(folder, status['checkpoint'], expected)
    if (status['binding'] != expected or status['counter'] != payload['counter']
            or status['log_head'] != payload['log_head']):
        raise ValueError('Committed status/model/log frontier mismatch')
    return payload, metadata


def publish(folder, manifest, cell, session, previous, *, state, reason=None):
    """Only status.json commits a generation. Orphan files never become resumed logs."""
    folder = Path(folder)
    started = time.perf_counter()
    before = previous['counter'] if previous else 0
    head = previous['log_head'] if previous else None
    rows = session.pending_rows
    if session.interactions-before != len(rows):
        raise ValueError('Pending rows/frontier mismatch')
    if previous and not rows:
        # A point has one checkpoint even if its administrative status changes.
        # Independent evaluation may already be bound to those immutable bytes.
        session.phase_seconds['checkpoint_logs'] += time.perf_counter()-started
        frontier = {**previous, 'state': state, 'reason': reason, 'metrics': session.metrics()}
        checked_write(folder/'status.json', frontier)
        return frontier
    if rows:
        header = {'schema': 'pong_log_segment_v2', 'start': before+1,
                  'end': session.interactions, 'previous': head}
        raw = b'\n'.join(json.dumps(v, sort_keys=True, allow_nan=False).encode() for v in [header, *rows])+b'\n'
        sha = bytes_hash(raw)
        path = folder/'logs'/f'{session.interactions}-{sha}.jsonl'
        if not path.exists():
            atomic_bytes(path, raw, replace=False)
        head = {'path': str(path.relative_to(folder)), 'sha256': sha, 'end': session.interactions}
    expected = binding(manifest, cell)
    # Cache is operational only; do not persist its materialized predictions.
    if hasattr(session.agent, 'cache'):
        session.agent.cache.clear()
    payload = {'binding': expected, 'counter': session.interactions, 'log_head': head,
               'session': session.state_dict()}
    raw = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    sha = bytes_hash(raw)
    path = folder/'checkpoints'/f'{session.interactions}-{sha}.pkl'
    ref = {'path': str(path.relative_to(folder)), 'sha256': sha}
    if not path.exists():
        atomic_bytes(path, raw, replace=False)
    metadata = {'checkpoint': ref, 'counter': session.interactions, 'log_head': head,
                'binding': expected, 'previous': previous['checkpoint'] if previous else None,
                'training_active_seconds': sum(session.phase_seconds.values())}
    if not Path(str(path)+'.json').exists():
        checked_write(str(path)+'.json', metadata, replace=False)
    frontier = {'binding': expected, 'counter': session.interactions, 'checkpoint': ref,
                'log_head': head, 'state': state, 'reason': reason, 'metrics': session.metrics()}
    # Full checkpoint time is in frontier metrics; next checkpoint also persists cumulative time.
    elapsed = time.perf_counter()-started
    session.phase_seconds['checkpoint_logs'] += elapsed
    frontier['metrics']['phase_seconds']['checkpoint_logs'] = session.phase_seconds['checkpoint_logs']
    checked_write(folder/'status.json', frontier)
    session.pending_rows = []
    return frontier


def committed_checkpoints(folder, status):
    refs = []
    seen = set()
    ref = status['checkpoint']
    while ref:
        meta = checkpoint_metadata(folder, ref)
        if ref['sha256'] in seen:
            raise ValueError('Checkpoint chain cycle')
        seen.add(ref['sha256'])
        refs.append(ref)
        ref = meta['previous']
    return refs


def inventory(folder, status, required_points=()):
    folder = Path(folder)
    refs = committed_checkpoints(folder, status)
    protected = {r['path'] for r in refs[:2]}
    points = set(required_points)
    for ref in refs:
        meta = checkpoint_metadata(folder, ref)
        if meta['counter'] in points:
            protected.add(ref['path'])
    # All evaluated checkpoint hashes and unfinished evaluation bindings remain protected.
    for path in (folder/'evaluations').rglob('identity.json'):
        evaluated = checked_read(path)['checkpoint']
        protected.add(evaluated['path'])
    confirmed_logs = {r['path'] for r in segment_refs(folder, status['log_head'])}
    by_path = {r['path']: r for r in refs}
    artifacts = []
    for path in sorted((folder/'checkpoints').glob('*.pkl')):
        relative = str(path.relative_to(folder))
        ref = by_path.get(relative)
        if ref:
            checkpoint_metadata(folder, ref)
        artifacts.append({'path': relative, 'bytes': path.stat().st_size,
                          'confirmed': ref is not None, 'protected': relative in protected,
                          'removable': relative not in protected})
    missing = [p for p in protected if not (folder/p).exists()]
    if missing:
        raise ValueError(f'Required retained checkpoint missing: {missing}')
    return {'schema': 'pong_retention_v2', 'checkpoint_bytes': sum(a['bytes'] for a in artifacts),
            'log_bytes': sum((folder/p).stat().st_size for p in confirmed_logs),
            'unconfirmed_logs': [str(p.relative_to(folder)) for p in (folder/'logs').glob('*.jsonl')
                                 if str(p.relative_to(folder)) not in confirmed_logs],
            'artifacts': artifacts}


def cleanup(folder, status, required_points):
    """Explicit blob cleanup only. Keep metadata chains and every confirmed log."""
    plan = inventory(folder, status, required_points)
    # Verify protected references first, without rereading every removable large blob.
    for item in plan['artifacts']:
        if item['protected']:
            ref = checked_read(Path(folder)/(item['path']+'.json'))['checkpoint']
            verified_bytes(folder, ref)
    for item in plan['artifacts']:
        if item['removable']:
            (Path(folder)/item['path']).unlink()
    for relative in plan['unconfirmed_logs']:
        (Path(folder)/relative).unlink()
    return plan
