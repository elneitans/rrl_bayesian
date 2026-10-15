"""V2 work units. Checksummed atomic commits, single writers and exact episodes."""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import tempfile
import time

import numpy as np

SCHEMA = 'pong_analysis_unit_v2'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def bytes_hash(value):
    return hashlib.sha256(value).hexdigest()


def array_hash(value):
    value = np.ascontiguousarray(value)
    return digest({'shape': value.shape, 'dtype': str(value.dtype), 'bytes': bytes_hash(value.tobytes())})


def atomic_bytes(path, data, *, replace=True):
    """Publish only flushed bytes; preserve the previous commit on any failure."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.' + path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        if replace:
            os.replace(name, path)
        else:
            os.link(name, path)  # exclusive atomic publication, even with concurrent writers
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        Path(name).unlink(missing_ok=True)


def atomic_json(path, value, *, replace=True):
    atomic_bytes(path, (json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n').encode(),
                 replace=replace)


def checked_write(path, value, *, replace=True):
    atomic_json(path, {'schema': SCHEMA, 'payload': value, 'sha256': digest(value)}, replace=replace)


def checked_read(path):
    try:
        value = json.loads(Path(path).read_text())
        if value['schema'] != SCHEMA or value['sha256'] != digest(value['payload']):
            raise ValueError('checksum/schema mismatch')
        return value['payload']
    except (KeyError, TypeError, json.JSONDecodeError, UnicodeError) as exc:
        raise ValueError(f'Corrupt analysis JSON: {path}') from exc


@contextmanager
def unit_lock(folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    # Never unlink the lock inode: waiting/new processes must see the same inode.
    with (folder / '.lock').open('a+b') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f'Concurrent writer for analysis unit: {folder}') from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def identity(manifest, *, pipeline, stage, method, dataset, variant, smoke, seeds,
             effective, hashes, evaluation=None, mcmc=None):
    """Operational time extensions are deliberately outside experimental identity."""
    from experiments.pong_analysis_common import source_versions
    return {'schema': SCHEMA, 'protocol': manifest['pilot']['pilot_protocol_sha256'],
            'analysis_config': digest(manifest['config']), 'runtime': source_versions(),
            'input_runtime': manifest['runtime'],
            'pipeline': pipeline, 'stage': stage, 'method': method, 'dataset': dataset,
            'variant': variant, 'mode': 'smoke' if smoke else 'full', 'seeds': list(seeds),
            'effective': effective, 'hashes': hashes, 'code': manifest['source_hashes'],
            'evaluation': evaluation, 'mcmc': mcmc}


def mode_root(analysis, pipeline, stage, smoke):
    return Path(analysis) / ('smoke' if smoke else 'full') / pipeline / stage


class WorkUnit:
    """All methods require the caller to hold unit_lock for this directory."""
    def __init__(self, folder, expected):
        # JSON represents tuples as arrays and integer keys as strings. Compare
        # the canonical persisted identity, not Python container conventions.
        expected = json.loads(json.dumps(expected, sort_keys=True, allow_nan=False))
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        path = self.folder / 'identity.json'
        if path.exists():
            if checked_read(path) != expected:
                raise ValueError(f'Analysis unit identity mismatch: {self.folder}')
        else:
            checked_write(path, expected, replace=False)
        self.expected = expected
        self.identity_hash = digest(expected)

    def load(self):
        path = self.folder / 'state.json'
        if not path.exists():
            return None
        state = checked_read(path)
        if (state.get('identity_sha256') != self.identity_hash or
                state.get('status') not in ('completed', 'partial', 'ready')):
            raise ValueError('Invalid work unit state/identity')
        for ref in state.get('artifacts', {}).values():
            self.read_blob(ref)  # validate all referenced bytes before accepting status
        return state

    def read_blob(self, ref):
        if ref['path'] != ref['sha256'] + '.blob':
            raise ValueError('Invalid work unit artifact reference')
        raw = (self.folder / 'objects' / ref['path']).read_bytes()
        if bytes_hash(raw) != ref['sha256']:
            raise ValueError('Analysis artifact checksum mismatch')
        return raw

    def commit(self, status, *, result=None, artifacts=None, reason=None):
        refs = {}
        for key, raw in (artifacts or {}).items():
            sha = bytes_hash(raw)
            path = self.folder / 'objects' / (sha + '.blob')
            if path.exists():
                if bytes_hash(path.read_bytes()) != sha:
                    raise ValueError('Corrupt content-addressed object')
            else:
                atomic_bytes(path, raw, replace=False)
            refs[key] = {'path': path.name, 'sha256': sha}
        state = {'identity_sha256': self.identity_hash, 'status': status, 'reason': reason,
                 'result': result, 'artifacts': refs}
        state = json.loads(json.dumps(state, sort_keys=True, allow_nan=False))
        checked_write(self.folder / 'state.json', state)
        # Keep only the current generation; interrupted cleanup is harmless.
        live = {ref['path'] for ref in refs.values()}
        for old in (self.folder / 'objects').glob('*.blob'):
            if old.name not in live:
                old.unlink()
        return state

    def pickle(self, state, key):
        return pickle.loads(self.read_blob(state['artifacts'][key]))  # trusted local files only


class Budget:
    """Cumulative operational clock. An extension token is applied only once."""
    def __init__(self, folder, seconds, *, extend_seconds=0., extension_id=None):
        self.path = Path(folder) / 'budget.json'
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError('Budget must be finite and positive')
        if (not math.isfinite(extend_seconds) or extend_seconds < 0 or
                (bool(extend_seconds) != bool(extension_id))):
            raise ValueError('Time extension requires positive seconds and an extension ID')
        self.value = checked_read(self.path) if self.path.exists() else {
            'base_seconds': seconds, 'elapsed_seconds': 0., 'extensions': []}
        if self.value['base_seconds'] != seconds:
            raise ValueError('Changed base budget; use an explicit extension')
        if extension_id:
            previous = next((v for v in self.value['extensions'] if v['id'] == extension_id), None)
            if previous is not None and previous['seconds'] != extend_seconds:
                raise ValueError('Extension ID already used with another amount')
            if previous is None:
                self.value['extensions'].append({'id': extension_id, 'seconds': extend_seconds,
                                                 'recorded_at_unix': time.time()})
        checked_write(self.path, self.value)
        self.last = time.monotonic()

    @property
    def remaining(self):
        limit = self.value['base_seconds'] + sum(v['seconds'] for v in self.value['extensions'])
        return max(0., limit - self.value['elapsed_seconds'] - (time.monotonic()-self.last))

    def charge(self):
        now = time.monotonic()
        self.value['elapsed_seconds'] += now - self.last
        self.last = now
        checked_write(self.path, self.value)


def extension_arguments(parser):
    parser.add_argument('--extend-seconds', type=float, default=0.)
    parser.add_argument('--extension-id')


def episode(factory, controller, folder, expected, *, seed, horizon, seconds,
            rng_seed, extend_seconds=0., extension_id=None, decision_limit=None,
            administrative_horizon=False, checkpoint_every=64, stop=None, pong_points=True):
    """Controller implements action(state, env, rng); its state is checkpointed.

    decision_limit is an administrative per-invocation limit for bounded tests.
    It never sets terminal/truncation. Environment snapshot includes relation history.
    """
    if type(checkpoint_every) is not int or checkpoint_every < 1:
        raise ValueError('Positive checkpoint interval required')
    with unit_lock(folder):
        unit = WorkUnit(folder, expected)
        saved = unit.load()
        if saved is not None and saved['status'] == 'completed':
            return saved['result']
        budget = Budget(folder, seconds, extend_seconds=extend_seconds, extension_id=extension_id)
        env = factory()
        try:
            if saved is not None:
                progress = unit.pickle(saved, 'progress')
                env.restore(progress['environment'])
                state, controller, rng, row, unique = (progress[k] for k in
                    ('state', 'controller', 'rng', 'row', 'unique'))
            else:
                state = env.reset(seed=seed)
                rng = np.random.default_rng(rng_seed)
                row = {'seed': seed, 'status': 'partial', 'steps': 0, 'raw_return': 0.,
                       'points_won': 0., 'points_lost': 0., 'actions': {},
                       'terminated': False, 'truncated': False}
                unique = set()

            def checkpoint(reason):
                row.update(status='partial', reason=reason, return_scope='administrative_partial',
                           unique_encoded_states=len(unique))
                payload = {'environment': env.snapshot(), 'state': state, 'controller': controller,
                           'rng': rng, 'row': row, 'unique': unique}
                unit.commit('partial', result=row, reason=reason,
                            artifacts={'progress': pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)})
                budget.charge()

            # Durable initial frontier; never need to concatenate independent attempts.
            if saved is None:
                checkpoint('ready')
            invocation_steps = 0
            while row['steps'] < horizon:
                if (budget.remaining <= 0 or (decision_limit is not None and invocation_steps >= decision_limit)
                        or (stop is not None and stop.requested)):
                    checkpoint('interrupted_signal' if stop is not None and stop.requested else
                               'time_budget' if budget.remaining <= 0 else 'administrative_pause')
                    return row
                batch = env.encoder().transform([state])
                unique.add(batch.values.tobytes() + batch.observed.tobytes())
                action = int(controller.action(state, env, rng))
                before = state
                next_state, learning_reward, reward, terminal, truncated = env.step(action)
                if hasattr(controller, 'transition'):
                    controller.transition(before, action, next_state, learning_reward, reward,
                                          terminal, truncated, env)
                if hasattr(controller, 'after_step'):
                    controller.after_step(reward)
                state = next_state
                row['steps'] += 1
                invocation_steps += 1
                row['raw_return'] += reward
                if pong_points:
                    row['points_won'] += max(reward, 0.)
                    row['points_lost'] += max(-reward, 0.)
                else:
                    row.pop('points_won',None)
                    row.pop('points_lost',None)
                name = env.actions[action]
                row['actions'][name] = row['actions'].get(name, 0) + 1
                row.update(terminated=bool(terminal), truncated=bool(truncated))
                if terminal or truncated or row['steps'] == horizon:
                    row.update(status='completed', reason=None,
                               truncated=bool(truncated or (row['steps'] == horizon and not terminal and not administrative_horizon)),
                               return_scope=('complete_game' if terminal else 'administrative_quota'
                                             if administrative_horizon and not truncated else 'horizon_limited'),
                               unique_encoded_states=len(unique))
                    if hasattr(controller, 'log'):
                        row['decision_log'] = controller.log
                    if hasattr(controller, 'records'):
                        row['transitions'] = controller.records
                    if administrative_horizon and not terminal and not truncated:
                        row['reason'] = 'decision_quota'
                    budget.charge()
                    row['seconds'] = budget.value['elapsed_seconds']
                    unit.commit('completed', result=row)
                    return row
                if row['steps'] % checkpoint_every == 0:
                    checkpoint('in_progress')
            raise ValueError('Partial episode checkpoint beyond its horizon')
        finally:
            # If step/action raises, the previous committed frontier remains valid.
            budget.charge()
            env.close()
