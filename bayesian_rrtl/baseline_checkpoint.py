"""Common baseline checkpoint v1. Pickle is for trusted local artifacts only.

Checksums detect corruption, not malicious pickle. No H0/H5 format is accepted.
Validate contracts and the complete collector before restoring environmental state.
"""

import copy
from dataclasses import asdict
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import pickle
import platform
import tempfile

import numpy as np

from .baseline import CorrectedNode
from .baseline_adapter import LEARNER_ID, BaselineLearnerConfig, RelationalBaselineAgent
from .baseline_training import BaselineTrainingSession
from .comparison import ComparisonPolicyConfig, PongEnvironmentFactory, _validate_environment
from .atari import AtariConfig
from .encoding import CATEGORIES, RelationKey, RelationalEncoder
from relational_regresion_tree import Node

MAGIC = b'RRTL-CORRECTED-COMMON-CHECKPOINT-1\n'
SCHEMA_VERSION = 1
COLLECTOR_FIELDS = ('state', 'done', 'episode_id', 'step_id', 'episode_return', 'raw_episode_return',
                    'history', 'episode_history', 'training_seeds')
SOURCES = ('relational_regresion_tree.py', 'preprocessing.py', 'bayesian_rrtl/baseline.py',
           'bayesian_rrtl/baseline_adapter.py', 'bayesian_rrtl/baseline_training.py',
           'bayesian_rrtl/baseline_checkpoint.py', 'bayesian_rrtl/comparison.py',
           'bayesian_rrtl/targets.py', 'bayesian_rrtl/encoding.py', 'bayesian_rrtl/atari.py',
           'bayesian_rrtl/objects.py', 'bayesian_rrtl/game_registry.py',
           'bayesian_rrtl/games/__init__.py', 'bayesian_rrtl/games/pong.py',
           'bayesian_rrtl/ocatari_backend.py')


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _hash(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _runtime():
    root = Path(__file__).resolve().parents[1]
    return {'python': platform.python_version(),
            'packages': {name: version(name) for name in ('numpy', 'scipy', 'gymnasium', 'ale-py')},
            'sources': {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in SOURCES}}


def _environment_contract(environment, snapshot):
    return {'config': environment.config.resolved_dict() if hasattr(environment, 'config') else None,
            'runtime': environment.runtime_manifest(), 'snapshot_kind': snapshot['kind']}


def _rng(state):
    rng = np.random.default_rng()
    try:
        rng.bit_generator.state = copy.deepcopy(state)
    except (TypeError, ValueError, KeyError) as exc:
        raise ValueError('Invalid saved RNG') from exc
    return rng


def _validate_trees(agent):
    seen, splits = set(), 0
    def visit(node, action, refinements, parent, depth):
        nonlocal splits
        _require(type(node) in (Node, CorrectedNode) and id(node) not in seen, 'Invalid tree node/cycle')
        seen.add(id(node))
        _require(node.parent is parent and node.depth == depth and node.root_action == action,
                 'Tree parent/action/depth mismatch')
        _require(node.refinements == list(refinements) and list(node.refinements_stats) == list(refinements),
                 'Tree candidate/statistics catalog mismatch')
        _require(node.valid_actions == list(agent.actions) and math.isfinite(node.q_value), 'Invalid tree Q/actions')
        for name in ('max_depth', 'min_sample_size', 'significance_level', 'best_literal_criteria'):
            _require(getattr(node, name) == getattr(agent.config, name), 'Tree configuration mismatch')
        for literal, groups in node.refinements_stats.items():
            _require(set(groups) == {'Total', *CATEGORIES[literal.type]}, 'Invalid statistic categories')
            for stats in groups.values():
                _require(set(stats) == {'n', 'mu', 'J'} and type(stats['n']) is int
                         and 0 <= stats['n'] <= agent.interactions
                         and math.isfinite(stats['mu']) and math.isfinite(stats['J']), 'Invalid split statistics')
            _require(groups['Total']['n'] == sum(groups[k]['n'] for k in CATEGORIES[literal.type]),
                     'Inconsistent split sample counters')
        if node.literal is None:
            _require(not node.children and all(getattr(node, name, None) is None for name in
                     ('more_branch', 'same_branch', 'less_branch', 'true_branch', 'false_branch')),
                     'Leaf has children')
            return
        _require(node.literal in refinements and depth < agent.config.max_depth, 'Invalid splitting literal/depth')
        splits += 1
        names = ('more', 'same', 'less') if node.literal.type == 'comparative' else ('true', 'false')
        children = [getattr(node, name + '_branch', None) for name in names]
        _require(node.children == children and all(child is not None for child in children), 'Invalid tree branches')
        remaining = [lit for lit in refinements if lit != node.literal]
        for child in children:
            visit(child, action, remaining, node, depth + 1)
    for action in agent.actions:
        visit(agent.root_nodes[action], action, agent.literals, None, 0)
    _require(splits == agent.n_splits, 'Tree/split counter mismatch')


def _validate_collector(agent, policy, saved, snapshot):
    _require(set(saved) == {*COLLECTOR_FIELDS, 'environment_rng', 'exploration_rng'}, 'Collector fields mismatch')
    _require(type(saved['done']) is bool and saved['done'] == snapshot['done'], 'Collector/environment done mismatch')
    for name in ('episode_id', 'step_id'):
        _require(type(saved[name]) is int, 'Invalid collector counters')
    _require(len(saved['history']) == agent.interactions, 'Collector/interaction counter mismatch')
    _require(agent.query_counts['update']['queries'] == agent.interactions, 'Update query counter mismatch')
    _rng(saved['exploration_rng'])
    environment_rng = np.random.default_rng(np.random.SeedSequence(agent.config.seed).spawn(2)[0])
    reserved = set(policy.validation_seeds + policy.test_seeds)
    for seed in saved['training_seeds']:
        candidate = int(environment_rng.integers(0, 2**32))
        while candidate in reserved:
            candidate = int(environment_rng.integers(0, 2**32))
        _require(type(seed) is int and candidate == seed, 'Training seeds differ from episode RNG stream')
    _require(environment_rng.bit_generator.state == _rng(saved['environment_rng']).bit_generator.state,
             'Episode RNG/seed counter mismatch')
    episode, step, total, raw_total, done, split_count = -1, 0, 0., 0., True, 0
    ended = []
    for index, row in enumerate(saved['history']):
        if done:
            episode += 1
            step, total, raw_total = 0, 0., 0.
        _require(episode < len(saved['training_seeds']), 'Missing episode seed')
        _require(row['transition_id'] == index and row['interaction'] == index + 1
                 and row['episode_id'] == episode and row['step_id'] == step
                 and row['episode_seed'] == saved['training_seeds'][episode], 'History IDs/seeds mismatch')
        _require(type(row['action']) is int and 0 <= row['action'] < len(agent.actions), 'History action mismatch')
        _require(all(type(row[name]) is bool for name in ('terminated', 'truncated', 'episode_end')),
                 'Invalid history flags')
        _require(math.isfinite(row['reward']) and math.isfinite(row['raw_reward']), 'Invalid history reward')
        total += row['reward']
        raw_total += row['raw_reward']
        step += 1
        done = row['terminated'] or row['truncated']
        scope = 'complete_game' if row['terminated'] else 'horizon_limited' if row['truncated'] else 'partial'
        _require(row['episode_end'] == done and row['return_scope'] == scope
                 and row['epsilon'] == policy.epsilon(index) and row['episode_return'] == total
                 and row['raw_episode_return'] == raw_total, 'History return/policy mismatch')
        specific = row['learner_metrics']
        _require(specific['kind'] == LEARNER_ID and specific['split_delta'] in (0, 1), 'Invalid learner log')
        split_count += specific['split_delta']
        _require(specific['splits'] == split_count, 'History split counter mismatch')
        if done:
            ended.append(BaselineTrainingSession._episode_row(row))
    _require((saved['episode_id'], saved['step_id'], saved['episode_return'], saved['raw_episode_return'], saved['done'])
             == (episode, step, total, raw_total, done), 'Current episode/collector mismatch')
    _require(len(saved['training_seeds']) == episode + 1 and saved['episode_history'] == ended,
             'Episode history mismatch')
    _require(split_count == agent.n_splits, 'History/tree split mismatch')
    if agent.interactions:
        agent.encoder.transform([saved['state']])
    else:
        _require(saved['state'] is None, 'Unstarted collector has an observation')


def _rebuild(payload, *, expected_config=None, expected_policy=None, expected_environment=None):
    _require(set(payload) == {'schema_version', 'learner', 'contract', 'contract_hash', 'learner_state',
                             'collector', 'environment_snapshot'}, 'Checkpoint fields mismatch')
    _require(payload['schema_version'] == SCHEMA_VERSION and payload['learner'] == LEARNER_ID,
             'Unsupported checkpoint version/learner')
    contract = payload['contract']
    _require(_hash(contract) == payload['contract_hash'], 'Checkpoint contract hash mismatch')
    _require(contract['runtime'] == _runtime(), 'Checkpoint source/runtime versions mismatch')
    config = BaselineLearnerConfig(**contract['learner']['config'])
    policy = ComparisonPolicyConfig(**contract['policy'])
    policy.validate_seed(config.seed)
    _require(expected_config is None or config == expected_config, 'Expected learner config mismatch')
    _require(expected_policy is None or policy == expected_policy, 'Expected policy mismatch')
    _require(expected_environment is None or expected_environment.resolved_dict() == contract['environment']['config'],
             'Expected environment mismatch')
    encoder = RelationalEncoder.from_dict(contract['learner']['encoder'])
    source = contract['learner']['catalog_source']
    _require(source in ('environment_encoder', 'explicit_historical_order'), 'Unknown candidate catalog source')
    ordered = ([RelationKey(**key) for key in contract['learner']['ordered_catalog']]
               if source == 'explicit_historical_order' else None)
    agent = RelationalBaselineAgent(contract['learner']['actions'], encoder, config, ordered_catalog=ordered)
    _require(agent.contract() == contract['learner'], 'Learner contract mismatch')
    _require(payload['learner_state']['contract'] == agent.contract(), 'Saved learner contract mismatch')
    agent.load_state_dict(payload['learner_state'])
    _validate_trees(agent)
    _validate_collector(agent, policy, payload['collector'], payload['environment_snapshot'])
    return agent, policy


def _validate_environment_snapshot(environment, contract, snapshot, agent):
    _validate_environment(agent, environment)
    _require(_environment_contract(environment, snapshot) == contract, 'Environment semantics/runtime mismatch')
    environment.validate_snapshot(snapshot)
    if snapshot['kind'] == 'atari_relational_v1':
        _require(snapshot['versions'] == {name: version(name) for name in ('gymnasium', 'ale-py')},
                 'Visual snapshot versions mismatch')
        _require([type(w).__name__ for w in environment._wrappers()] == [name for name, _ in snapshot['wrappers']],
                 'Visual wrapper stack mismatch')
        _require(environment.extractor.snapshot() == snapshot['extractor'], 'Visual extractor mismatch')
        _rng(snapshot['numpy_rng'])


def save_baseline_checkpoint(session, path):
    _require(type(session) is BaselineTrainingSession and session._checkpoint_ready,
             'Checkpoint requires a completed baseline update')
    snapshot = session.environment.snapshot()
    collector = {key: copy.deepcopy(getattr(session, key)) for key in COLLECTOR_FIELDS}
    collector.update(environment_rng=copy.deepcopy(session.environment_rng.bit_generator.state),
                     exploration_rng=copy.deepcopy(session.exploration_rng.bit_generator.state))
    contract = {'learner': session.agent.contract(), 'policy': asdict(session.policy_config),
                'environment': _environment_contract(session.environment, snapshot), 'runtime': _runtime()}
    payload = {'schema_version': SCHEMA_VERSION, 'learner': LEARNER_ID, 'contract': contract,
               'contract_hash': _hash(contract), 'learner_state': session.agent.state_dict(),
               'collector': collector, 'environment_snapshot': snapshot}
    # The same checks apply at save and load, including cross-component counters.
    agent, _ = _rebuild(payload)
    _validate_environment_snapshot(session.environment, contract['environment'], snapshot, agent)
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


def load_baseline_checkpoint(path, environment=None, *, expected_config=None, expected_policy=None,
                             expected_environment=None, environment_factory=None):
    """Validate first; allocate/restore a new Atari environment unless one is supplied."""
    with Path(path).open('rb') as handle:
        _require(handle.readline() == MAGIC, 'Unsupported baseline checkpoint format')
        digest, data = handle.readline().strip(), handle.read()
    _require(hashlib.sha256(data).hexdigest().encode() == digest, 'Baseline checkpoint checksum mismatch')
    payload = pickle.loads(data)
    try:
        agent, policy = _rebuild(payload, expected_config=expected_config, expected_policy=expected_policy,
                                 expected_environment=expected_environment)
    except (KeyError, TypeError, AttributeError, IndexError) as exc:
        raise ValueError('Malformed baseline checkpoint state') from exc
    env_contract = payload['contract']['environment']
    factory = environment_factory
    if factory is None and env_contract['config'] is not None:
        factory = PongEnvironmentFactory(AtariConfig(**env_contract['config']))
    owned = environment is None
    if owned:
        _require(factory is not None, 'Non-Atari checkpoint requires an explicit environment/factory')
        environment = factory()
    try:
        snapshot = payload['environment_snapshot']
        _validate_environment_snapshot(environment, env_contract, snapshot, agent)
        # Initialize an empty collector without discarding restored statistics.
        fresh = RelationalBaselineAgent(agent.actions, agent.encoder, agent.config,
                    ordered_catalog=agent.catalog if agent.catalog_source == 'explicit_historical_order' else None)
        session = BaselineTrainingSession(fresh, environment, policy, environment_factory=factory)
        session.agent = agent
        for key in COLLECTOR_FIELDS:
            setattr(session, key, copy.deepcopy(payload['collector'][key]))
        session.environment_rng = _rng(payload['collector']['environment_rng'])
        session.exploration_rng = _rng(payload['collector']['exploration_rng'])
        environment.restore(snapshot)
        return session
    except Exception:
        if owned:
            environment.close()
        raise


def evaluate_baseline_checkpoint(path, seeds, *, horizon=None, environment_factory=None, **load_options):
    """Final evaluation loads the saved artifact; never saves over it."""
    session = load_baseline_checkpoint(path, environment_factory=environment_factory, **load_options)
    try:
        return session.evaluate(seeds, horizon=horizon, environment_factory=environment_factory)
    finally:
        session.environment.close()
