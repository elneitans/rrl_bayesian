"""Conditional OCAtari behavior-assisted collection on new development seeds."""
import argparse
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np

from bayesian_rrtl.atari import AtariConfig
from bayesian_rrtl.q_config import BayesianQConfig
from bayesian_rrtl.replay import ReplayBuffer
from experiments.evaluate_pong_heuristics import ReactivePolicy, object_frame
from experiments.freeze_pong_analysis_data import fixed_transitions
from experiments.pong_analysis_common import config, load_manifest, read, source_path, write_new


def collect(analysis):
    manifest = load_manifest(analysis)
    cfg = manifest['config']
    report = read(Path(analysis) / 'ocatari/heuristics.json')
    episodes = report['policies']['object_reactive']
    seeds = cfg['development_seeds']
    if (len(episodes) != 20 or any(episodes[str(s)]['status'] != 'completed' for s in seeds)
            or report['paired_vs_random']['object_reactive']['mean_difference'] <= 0):
        raise ValueError('Behavior-assisted collection needs the predeclared successful heuristic gate')
    from bayesian_rrtl.comparison import PongEnvironmentFactory
    protocol = read(source_path(cfg['source_campaign']) / 'protocol.json')['protocol']
    env_config = AtariConfig(**protocol['environments']['ocatari'])
    factory = PongEnvironmentFactory(env_config)
    directions = read(Path(analysis) / 'ocatari/action_probe.json')
    learner_config = BayesianQConfig(**next(cell['config'] for cell in
        read(source_path(cfg['source_campaign']) / 'protocol.json')['cells'] if cell['id'] == 'ocatari/bart/71'))
    from bayesian_rrtl.games.pong import PongRelationsV1
    encoder = PongRelationsV1().encoder()
    replay = ReplayBuffer(cfg['assisted_max_decisions'])
    rng = np.random.default_rng(cfg['mcmc_seed'] + 10000)
    summaries = []
    quota = max(1, cfg['assisted_max_decisions'] // len(cfg['assisted_collection_seeds']))
    for episode_id, seed in enumerate(cfg['assisted_collection_seeds']):
        env = factory()
        policy = ReactivePolicy(directions['up'], directions['down'], env.actions)
        started = time.monotonic()
        reward_sum = 0.
        try:
            state = env.reset(seed=seed)
            for step_id in range(quota):
                if time.monotonic()-started >= cfg['max_seconds_per_episode']:
                    summaries.append({'seed': seed, 'reason': 'time_budget', 'steps': step_id})
                    break
                if rng.random() < cfg['assisted_random_fraction']:
                    action = int(rng.integers(len(env.actions)))
                else:
                    action = policy.object_action(object_frame(env))
                next_state, reward, raw, terminated, truncated = env.step(action)
                replay.append(state=encoder.transform([state]), action=action, reward=float(reward),
                    raw_reward=float(raw), next_state=None if terminated else encoder.transform([next_state]),
                    terminated=terminated, truncated=truncated, episode_id=episode_id, step_id=step_id)
                reward_sum += raw
                state = next_state
                if terminated or truncated:
                    summaries.append({'seed': seed, 'reason': 'terminated' if terminated else 'truncated',
                                      'steps': step_id+1, 'raw_return': reward_sum})
                    break
            else:
                summaries.append({'seed': seed, 'reason': 'decision_quota', 'steps': quota,
                                  'raw_return': reward_sum})
        finally:
            env.close()
    agent = SimpleNamespace(replay=replay, encoder=encoder, config=learner_config)
    output = Path(analysis) / 'ocatari/assisted_fixed_transitions'
    result = fixed_transitions(agent, None, output, batch_per_action=cfg['fqi_batch_per_action'],
        fraction=cfg['fqi_validation_episode_fraction'], seed=cfg['mcmc_seed']+1,
        behavior='predeclared_50_percent_random_50_percent_object_reactive')
    write_new(output / 'collection.json', {'episodes': summaries, 'seeds': cfg['assisted_collection_seeds'],
        'random_fraction': cfg['assisted_random_fraction'],
        'development_heuristic_report': 'ocatari/heuristics.json',
        'notes': 'Distinct assisted behavior dataset, not a repeat of original pilot training.',
        'transitions_sha256': result['npz_sha256']})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', type=Path, required=True)
    args = parser.parse_args()
    collect(args.analysis)


if __name__ == '__main__':
    main()
