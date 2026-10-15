"""BART comparison collector persistence layered on the unchanged H5 payload."""
import copy

from .agent import BayesianQAgent
from .atari import AtariConfig
from .baseline_checkpoint import _runtime
from .checkpoint import load_checkpoint, save_checkpoint
from .comparison import BayesianComparisonSession, ComparisonPolicyConfig, PongEnvironmentFactory, _validate_environment


def save_session(session, path):
    if type(session.agent) is not BayesianQAgent:
        raise ValueError('Historic comparison checkpoint requires BART; use save_online_session for v2')
    if not session._checkpoint_ready:
        raise ValueError('Checkpoint requires a completed comparison update')
    collector = {k: copy.deepcopy(v) for k, v in vars(session).items()
                 if k not in ('agent', 'environment', 'environment_factory')}
    snapshot = session.environment.snapshot()
    state = {'kind': 'bart_comparison_collector_v1', 'collector': collector,
             'environment': snapshot, 'runtime_environment': session.environment.runtime_manifest(),
             'runtime': _runtime()}
    return save_checkpoint(session.agent, path, collector_state=state)


def load_session(path, *, expected_config=None):
    agent, saved = load_checkpoint(path, expected_config=expected_config)
    if (type(agent) is not BayesianQAgent or not isinstance(saved, dict)
            or saved.get('kind') != 'bart_comparison_collector_v1' or saved['runtime'] != _runtime()):
        raise ValueError('Incompatible BART comparison checkpoint')
    c = saved['collector']
    policy = c['policy_config']
    if not isinstance(policy, ComparisonPolicyConfig):
        raise ValueError('Invalid comparison policy')
    policy.validate_bart(agent.config)
    history = c['history']
    if (len(history) != agent.interactions or not c['_checkpoint_ready']
            or c['done'] != saved['environment']['done']
            or len(agent.training_seeds) != c['episode_id'] + 1):
        raise ValueError('Comparison collector counters mismatch')
    for i, row in enumerate(history):
        if (row['interaction'] != i + 1 or row['transition_id'] != i
                or row['episode_seed'] != agent.training_seeds[row['episode_id']]
                or row['epsilon'] != policy.epsilon(i)):
            raise ValueError('Comparison history mismatch')
    if history:
        last = history[-1]
        if (c['episode_id'] != last['episode_id'] or c['step_id'] != last['step_id'] + 1
                or c['episode_return'] != last['episode_return']
                or c['raw_episode_return'] != last['raw_episode_return']
                or c['done'] != last['episode_end']):
            raise ValueError('Current episode mismatch')
        agent.encode_state(c['state'])
    ended = [BayesianComparisonSession._episode_row(row) for row in history if row['episode_end']]
    if c['episode_history'] != ended:
        raise ValueError('Episode history mismatch')
    factory = PongEnvironmentFactory(AtariConfig(**saved['environment']['config']))
    env = factory()
    try:
        _validate_environment(agent, env)
        env.validate_snapshot(saved['environment'])
        if env.runtime_manifest() != saved['runtime_environment']:
            raise ValueError('Environment runtime mismatch')
        session = BayesianComparisonSession(BayesianQAgent(agent.config, agent.encoder), env, policy,
                                            environment_factory=factory)
        if set(c) != set(vars(session)) - {'agent', 'environment', 'environment_factory'}:
            raise ValueError('Comparison collector fields mismatch')
        vars(session).update(c)
        session.agent = agent
        env.restore(saved['environment'])
        return session
    except Exception:
        env.close()
        raise


def save_online_session(session, path):
    """Standalone v2 collector; legacy H5/BART payloads use save_session above."""
    import pickle
    from pathlib import Path
    from experiments.pong_analysis_runtime import atomic_bytes, bytes_hash, checked_write
    from .online.protocol import runtime, source_hashes
    raw = pickle.dumps(session.state_dict(), protocol=pickle.HIGHEST_PROTOCOL)
    path = Path(path)
    atomic_bytes(path, raw, replace=False)
    checked_write(str(path)+'.json', {'schema': 'pong_online_session_v2',
        'sha256': bytes_hash(raw), 'runtime': runtime(session.learner),
        'sources': source_hashes(), 'cell': session.cell}, replace=False)
    return path


def load_online_session(path, *, cell, environment):
    import pickle
    from pathlib import Path
    from experiments.pong_analysis_runtime import bytes_hash, checked_read, digest
    from .online.protocol import runtime, source_hashes
    from .online.session import OnlineSession
    meta = checked_read(str(path)+'.json')
    raw = Path(path).read_bytes()
    if (meta['schema'] != 'pong_online_session_v2' or meta['sha256'] != bytes_hash(raw)
            or meta['runtime'] != runtime(cell['learner']) or meta['sources'] != source_hashes()
            or digest(meta['cell']) != digest(cell)):
        raise ValueError('Incompatible online checkpoint')
    return OnlineSession.restore(pickle.loads(raw), cell, environment)


def checkpoint_format(path):
    """Recognize historical formats explicitly, without migrating or relaxing hashes."""
    from pathlib import Path
    from .checkpoint import MAGIC as bart_magic
    from .baseline_checkpoint import MAGIC as baseline_magic
    from experiments.pong_analysis_runtime import checked_read
    with Path(path).open('rb') as stream:
        prefix = stream.read(max(len(bart_magic), len(baseline_magic)))
    if prefix.startswith(bart_magic):
        return 'bart_historic_v1'
    if prefix.startswith(baseline_magic):
        return 'rrtl_common_historic_v1'
    sidecar = Path(str(path)+'.json')
    if sidecar.exists():
        schema = checked_read(sidecar).get('schema')
        if schema == 'pong_online_session_v2':
            return schema
        if 'binding' in checked_read(sidecar):
            return 'pong_progressive_checkpoint_v2'
    raise ValueError('Unknown checkpoint format')
