"""H6 Breakout runner: replaceable visual extraction, paired corrected baseline."""

import argparse
import cProfile
import csv
from dataclasses import replace
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import subprocess
import time

import numpy as np

from bayesian_rrtl.agent import BayesianQAgent
from bayesian_rrtl.atari import AtariConfig, AtariRelationalEnvironment, game_actions
from bayesian_rrtl.checkpoint import load_checkpoint
from bayesian_rrtl.profiling import profile_summary
from bayesian_rrtl.q_config import BayesianQConfig
from bayesian_rrtl.training import QTrainingSession

ROOT = Path(__file__).resolve().parent


def validate_pair(env_config, config):
    if config.actions != game_actions(env_config.game):
        raise ValueError('Agent actions differ from ALE action order')
    bounds = (-.9, 1.1) if env_config.game == 'Pong' else (-1., 1.)
    if (config.reward_min, config.reward_max) != bounds:
        raise ValueError('Reward bounds must match the game reward transformation')


def manifest(env_config, config, variant):
    sources = [Path(__file__), *sorted((ROOT / 'bayesian_rrtl').glob('*.py')),
               ROOT / 'preprocessing.py', ROOT / 'relational_regresion_tree.py']
    return {'variant': variant, 'environment': env_config.to_dict(), 'agent': config.to_dict(),
            'agent_config_hash': config.config_hash, 'python': platform.python_version(),
            'packages': {name: version(name) for name in ('numpy', 'scipy', 'gymnasium', 'ale-py', 'opencv-python')},
            'git_commit': subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True,
                                         text=True, check=True).stdout.strip(),
            'source_hashes': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
            'checkpoint_selection': 'final', 'interaction_unit': 'agent env.step, FIRE startup excluded',
            'comparison': 'Same extractor, reward, action handling, episode-seed stream and interaction budget; '
                          'different policies produce different trajectories. Legacy vs binary representation remains a confound.'}


def run_bayesian(env_config, config, output, steps, eval_split, resume=None):
    env = AtariRelationalEnvironment(env_config)
    try:
        session = (QTrainingSession.load(resume, env, expected_config=config) if resume else
                   QTrainingSession(BayesianQAgent(config, env.encoder()), env))
        profiler = cProfile.Profile()
        started = time.perf_counter()
        profiler.enable()
        session.run(steps)
        profiler.disable()
        seconds = time.perf_counter() - started
        stage_profile = profile_summary(profiler)
        stage_profile['extractor'] = env.profile
        profiler.dump_stats(str(output / 'training.prof'))
        checkpoint = session.save(output / 'final_checkpoint.rrtl')
        with (output / 'train.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(session.history[0]))
            writer.writeheader()
            writer.writerows(session.history)
        agent, _ = load_checkpoint(checkpoint)
        (output / 'fits.json').write_text(json.dumps(agent.fit_history, indent=2, allow_nan=False))
        (output / 'encoder.json').write_text(json.dumps(agent.encoder.to_dict(), indent=2))
        evaluation = []
        if eval_split != 'none':
            evaluation = agent.evaluate(lambda: AtariRelationalEnvironment(env_config),
                                        getattr(config, f'{eval_split}_seeds'))
            (output / f'{eval_split}.json').write_text(json.dumps(evaluation, indent=2))
        action_summary = {}
        if agent.last_fit:
            for dataset in agent.last_fit.datasets:
                posterior = agent.model.posteriors[dataset.action]
                if posterior is None or not len(dataset.batch):
                    continue
                start = time.perf_counter()
                predictions = posterior.predict_mean(dataset.batch)
                prediction_seconds = time.perf_counter() - start
                if not np.isfinite(predictions).all() or not np.isfinite(dataset.targets).all():
                    raise ValueError('Non-finite targets or Q predictions')
                retained = posterior.trace[posterior.burn_in:]
                action_summary[config.actions[dataset.action]] = {
                    'samples': len(dataset.batch), 'fit_seconds': posterior.elapsed_seconds,
                    'prediction_seconds': prediction_seconds, 'prediction_rows': len(dataset.batch),
                    'mean_total_leaves': float(np.mean([s.total_leaves for s in retained])),
                    'max_depth': max(s.max_depth for s in retained),
                    'mean_sigma2_normalized': float(np.mean(posterior.sigma2)),
                    'mean_residual_sum_squares_normalized': float(np.mean([s.residual_sum_squares for s in retained])),
                    'max_abs_q': float(np.abs(predictions).max()), 'moves': posterior.diagnostics()['moves'],
                    'targets_outside_scale': posterior.targets_outside_scale}
        return {'variant': 'rrtl_b', 'interactions': agent.interactions, 'additional_interactions': steps,
                'training_seconds': seconds, 'fits': agent.model.model_id, 'profile': stage_profile,
                'checkpoint_bytes': checkpoint.stat().st_size, 'actions': action_summary,
                'evaluation': evaluation, 'training_seeds': agent.training_seeds,
                'completed_episodes': sum(r['terminated'] or r['truncated'] for r in session.history),
                'finite_data': all(np.isfinite(r['reward']) and np.isfinite(r['raw_reward']) for r in session.history)}
    finally:
        env.close()


def run_corrected(env_config, config, output, steps, eval_split):
    from bayesian_rrtl.baseline import CorrectedRRLAgent
    from bayesian_rrtl.config import BaselineConfig
    # Match all shared experimental controls, keeping the corrected learner unchanged.
    baseline_config = BaselineConfig(game=env_config.game, model_version=env_config.representation,
        seed=config.seed, n_iterations=steps, gamma=config.gamma, epsilon_init=config.epsilon_init,
        epsilon_min=config.epsilon_min, epsilon_decay_steps=config.epsilon_decay_steps,
        bootstrap_on_truncation=config.bootstrap_on_truncation,
        resample_repeated_actions=config.resample_repeated_actions, action_buffer_capacity=config.action_buffer_capacity,
        include_incomplete_states=env_config.include_incomplete_states, frameskip=env_config.frameskip,
        repeat_action_probability=env_config.repeat_action_probability, max_episode_steps=env_config.max_episode_steps,
        validation_seeds=config.validation_seeds, test_seeds=config.test_seeds)
    agent = CorrectedRRLAgent(baseline_config, output)
    # H0 uses child 1 for environments, H5 uses child 0. Pair the episode-seed stream explicitly.
    agent.environment_rng = np.random.default_rng(np.random.SeedSequence(config.seed).spawn(3 + len(config.actions))[0])
    (output / 'baseline_config.json').write_text(json.dumps(baseline_config.to_dict(), indent=2))
    profiler = cProfile.Profile()
    started = time.perf_counter()
    profiler.enable()
    summary = agent.train()
    profiler.disable()
    seconds = time.perf_counter() - started
    profiler.dump_stats(str(output / 'training.prof'))
    agent.load_final_checkpoint(output / 'final_checkpoint.pkl')
    evaluation = []
    if eval_split != 'none':
        evaluation = agent.evaluate(getattr(config, f'{eval_split}_seeds'))
        (output / f'{eval_split}.json').write_text(json.dumps(evaluation, indent=2))
    rows = list(csv.DictReader((output / 'train.csv').open()))
    if len(rows) != steps:
        raise ValueError('Baseline interaction count mismatch')
    return {'variant': 'rrtl_corrected', 'interactions': steps, 'additional_interactions': steps,
            'training_seconds': seconds, 'profile': profile_summary(profiler), 'evaluation': evaluation,
            'training_seeds': agent.training_seeds, 'completed_episodes': summary['completed_episodes'],
            'finite_data': bool(np.isfinite([[float(r[k]) for k in ('Reward', 'Raw reward', 'New Q', 'TD target')]
                                             for r in rows]).all())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--config', type=Path)
    source.add_argument('--resume', type=Path)
    source.add_argument('--checkpoint', type=Path, help='Evaluate Bayesian checkpoint without training')
    parser.add_argument('--variant', choices=('bayesian', 'corrected'), default='bayesian')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--steps', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--eval-split', choices=('validation', 'test', 'none'), default='validation')
    args = parser.parse_args()
    if args.resume or args.checkpoint:
        if args.variant != 'bayesian' or args.seed is not None:
            parser.error('Checkpoint mode requires bayesian variant and its saved seed')
        saved_agent, collector = load_checkpoint(args.resume or args.checkpoint)
        if collector is None or collector['environment']['kind'] != 'atari_relational_v1':
            parser.error('Expected a full Atari checkpoint')
        env_config = AtariConfig(**collector['environment']['config'])
        config = saved_agent.config
        default_steps = 1024
    else:
        raw = json.loads((args.config or ROOT / 'configs/bayesian_breakout_pilot.json').read_text())
        env_config, config = AtariConfig(**raw['environment']), BayesianQConfig(**raw['agent'])
        config = replace(config, seed=args.seed) if args.seed is not None else config
        default_steps = raw['interactions']
    validate_pair(env_config, config)
    if args.checkpoint:
        if args.steps is not None or args.output is not None or args.eval_split == 'none':
            parser.error('Evaluation only accepts checkpoint and validation/test split')
        rows = saved_agent.evaluate(lambda: AtariRelationalEnvironment(env_config),
                                    getattr(config, f'{args.eval_split}_seeds'))
        destination = args.checkpoint.resolve().parent / f'{args.eval_split}.json'
        destination.write_text(json.dumps(rows, indent=2))
        print(json.dumps({'evaluation': str(destination)}))
        return
    steps = args.steps if args.steps is not None else default_steps
    if type(steps) is not int or steps < 1:
        parser.error('Interaction budget must be positive')
    if args.output is None:
        parser.error('Training requires --output with a new directory')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    info = manifest(env_config, config, args.variant)
    info['additional_interactions'] = steps
    info['resumed_from'] = str(args.resume.resolve()) if args.resume else None
    (output / 'manifest.json').write_text(json.dumps(info, indent=2))
    if args.variant == 'bayesian':
        summary = run_bayesian(env_config, config, output, steps, args.eval_split, args.resume)
    else:
        summary = run_corrected(env_config, config, output, steps, args.eval_split)
    if not summary['finite_data']:
        raise ValueError('Non-finite training data')
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    print(json.dumps({'output': str(output), 'seconds': summary['training_seconds'],
                      'interactions': summary['interactions']}), flush=True)


if __name__ == '__main__':
    main()
