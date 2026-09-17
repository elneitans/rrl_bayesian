"""Run H5 on the known-Q MDP; use adapters for other relational environments."""

import argparse
import csv
import json
from pathlib import Path
import platform

import numpy as np

from bayesian_rrtl.agent import BayesianQAgent
from bayesian_rrtl.checkpoint import load_checkpoint
from bayesian_rrtl.q_config import BayesianQConfig
from bayesian_rrtl.toy_mdp import TwoStepMDP
from bayesian_rrtl.training import QTrainingSession

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--config', type=Path)
    source.add_argument('--resume', type=Path, help='Resume full collector checkpoint into a new output directory')
    source.add_argument('--checkpoint', type=Path, help='Evaluate existing checkpoint without training')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--steps', type=int, help='Additional environment interactions (default: 600)')
    parser.add_argument('--eval-split', choices=('validation', 'test', 'none'), default='validation')
    args = parser.parse_args()
    if args.checkpoint:
        if args.steps is not None or args.output is not None or args.eval_split == 'none':
            parser.error('--checkpoint accepts only --eval-split validation/test')
        agent, _ = load_checkpoint(args.checkpoint)
        if agent.config.actions != TwoStepMDP.actions or agent.encoder.catalog_hash != TwoStepMDP.encoder().catalog_hash:
            parser.error('This runner requires a TwoStepMDP checkpoint')
        destination = args.checkpoint.resolve().parent / f'{args.eval_split}.json'
        rows = agent.evaluate(TwoStepMDP, getattr(agent.config, f'{args.eval_split}_seeds'))
        destination.write_text(json.dumps(rows, indent=2))
        print(json.dumps({'evaluation': str(destination), 'episodes': len(rows)}))
        return
    if args.steps is not None and args.steps < 1:
        parser.error('--steps must be positive')
    env = TwoStepMDP()
    if args.resume:
        session = QTrainingSession.load(args.resume, env)
    else:
        config = BayesianQConfig.load(args.config or ROOT / 'configs/bayesian_toy.json')
        if config.actions != env.actions:
            parser.error('This runner requires actions advance, exit in that order')
        session = QTrainingSession(BayesianQAgent(config, env.encoder()), env)
    output = (args.output or ROOT / 'results/h5_toy').resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = session.agent.config
    (output / 'manifest.json').write_text(json.dumps({
        'variant': 'rrtl_b', 'environment': 'two_step_v1', 'config': config.to_dict(),
        'config_hash': config.config_hash, 'encoder': session.agent.encoder.to_dict(),
        'scale': {'c': config.scale.c, 'W': config.scale.width},
        'python': platform.python_version(), 'numpy': np.__version__,
        'resumed_from': str(args.resume.resolve()) if args.resume else None,
        'checkpoint_selection': 'final', 'interaction_unit': 'environment.step'}, indent=2))
    try:
        session.run(args.steps if args.steps is not None else 600)
        checkpoint = session.save(output / 'final_checkpoint.rrtl')
        with (output / 'train.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(session.history[0]))
            writer.writeheader()
            writer.writerows(session.history)
        (output / 'fits.json').write_text(json.dumps(session.agent.fit_history, indent=2))
        # Evaluate the final artifact itself using independent environments.
        agent, _ = load_checkpoint(checkpoint)
        if args.eval_split != 'none':
            rows = agent.evaluate(TwoStepMDP, getattr(config, f'{args.eval_split}_seeds'))
            (output / f'{args.eval_split}.json').write_text(json.dumps(rows, indent=2))
        batch = agent.encoder.transform([env.state(0), env.state(1)])
        predicted, optimal = agent.model.predict(batch), env.optimal_q(config.gamma)
        summary = {'interactions': agent.interactions, 'fits': agent.model.model_id,
                   'q': predicted.tolist(), 'optimal_q': optimal.tolist(),
                   'max_abs_q_error': float(np.abs(predicted - optimal).max()),
                   'greedy_actions': predicted.argmax(axis=1).tolist(),
                   'checkpoint': str(checkpoint)}
        (output / 'summary.json').write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))
    finally:
        env.close()


if __name__ == '__main__':
    main()
