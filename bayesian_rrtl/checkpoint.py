"""Atomic versioned checkpoints for trusted local files only (pickle payload)."""

import hashlib
import json
import os
from pathlib import Path
import pickle
import platform
import tempfile

import numpy as np

from .agent import BayesianQAgent
from .encoding import RelationalEncoder
from .q_config import BayesianQConfig

MAGIC = b'RRTL-B-CHECKPOINT-1\n'
SCHEMA_VERSION = 1


def save_checkpoint(agent, path, *, collector_state=None):
    payload = {'learner': type(agent).__name__, 'schema_version': SCHEMA_VERSION, 'config': agent.config.to_dict(),
               'config_hash': agent.config.config_hash, 'encoder': agent.encoder.to_dict(),
               'state': {key: value for key, value in vars(agent).items() if key not in ('config', 'encoder')},
               'scale': {'c': agent.config.scale.c, 'W': agent.config.scale.width},
               'collector_state': collector_state,
               'runtime': {'python': platform.python_version(), 'numpy': np.__version__}}
    data = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + '.', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'wb') as handle:
            handle.write(MAGIC + hashlib.sha256(data).hexdigest().encode() + b'\n' + data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return path


def load_checkpoint(path, *, expected_config=None):
    """Return agent and optional collector snapshot; reject mismatches before use."""
    with Path(path).open('rb') as handle:
        if handle.readline() != MAGIC:
            raise ValueError('Unsupported checkpoint format')
        digest, data = handle.readline().strip(), handle.read()
    if hashlib.sha256(data).hexdigest().encode() != digest:
        raise ValueError('Checkpoint checksum mismatch')
    payload = pickle.loads(data)
    if payload['schema_version'] != SCHEMA_VERSION:
        raise ValueError('Unsupported checkpoint schema')
    config = BayesianQConfig(**payload['config'])
    # Verify the original serialized config before filling additive defaults.
    # H5 checkpoints omit warmup_interactions; their behavior is warmup=0.
    stored_hash = hashlib.sha256(json.dumps(payload['config'], sort_keys=True).encode()).hexdigest()
    if stored_hash != payload['config_hash'] or (
            expected_config is not None and expected_config.config_hash != config.config_hash):
        raise ValueError('Checkpoint config mismatch')
    encoder = RelationalEncoder.from_dict(payload['encoder'])
    from .controls import BatchTreeAgent, IncrementalBinaryAgent
    learners = {'BayesianQAgent': BayesianQAgent, 'BatchTreeAgent': BatchTreeAgent,
                'IncrementalBinaryAgent': IncrementalBinaryAgent}
    if payload.get('learner', 'BayesianQAgent') not in learners:
        raise ValueError('Unknown checkpoint learner')
    learner = learners[payload.get('learner', 'BayesianQAgent')]
    kwargs = ({'eta': payload['state']['eta'], 'split_min': payload['state']['split_min']}
              if learner is IncrementalBinaryAgent else {})
    agent = learner(config, encoder, **kwargs)
    if set(payload['state']) != set(vars(agent)) - {'config', 'encoder'}:
        raise ValueError('Checkpoint state fields mismatch')
    if payload['scale'] != {'c': config.scale.c, 'W': config.scale.width}:
        raise ValueError('Checkpoint scale mismatch')
    vars(agent).update(payload['state'])
    if agent.replay.capacity != config.replay_capacity or agent.replay.next_id != agent.interactions:
        raise ValueError('Checkpoint replay counters mismatch')
    for model in (agent.model, agent.target_model):
        if model.actions != config.actions or model.catalog_hash != encoder.catalog_hash:
            raise ValueError('Checkpoint model/encoder mismatch')
        for posterior in model.posteriors:
            if posterior is not None and (posterior.catalog_hash != encoder.catalog_hash or
                                           posterior.scale != config.scale):
                raise ValueError('Checkpoint posterior mismatch')
    # NumPy pickle does not preserve the readonly flag.
    for transition in agent.replay.transitions:
        for batch in (transition.state, transition.next_state):
            if batch is not None:
                batch.values.flags.writeable = batch.observed.flags.writeable = False
    if agent.last_fit is not None:
        for dataset in agent.last_fit.datasets:
            dataset.batch.values.flags.writeable = dataset.batch.observed.flags.writeable = False
            dataset.targets.flags.writeable = False
    return agent, payload['collector_state']
