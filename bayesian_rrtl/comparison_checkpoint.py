"""BART comparison collector persistence layered on the unchanged H5 payload."""
import copy

from .agent import BayesianQAgent
from .atari import AtariConfig
from .baseline_checkpoint import _runtime
from .checkpoint import load_checkpoint, save_checkpoint
from .comparison import BayesianComparisonSession, ComparisonPolicyConfig, PongEnvironmentFactory, _validate_environment


def save_session(session, path):
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
