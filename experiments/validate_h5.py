"""Reproducible H5 acceptance checks; no Atari or test-seed calibration."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np

from bayesian_rrtl.agent import BayesianQAgent
from bayesian_rrtl.q_config import BayesianQConfig
from bayesian_rrtl.toy_mdp import TwoStepMDP
from bayesian_rrtl.training import QTrainingSession


def validate(output):
    output.mkdir(parents=True, exist_ok=False)
    results = []
    base = BayesianQConfig.load(Path(__file__).resolve().parents[1] / 'configs/bayesian_toy.json')
    for seed in (11, 22, 33):
        started = time.perf_counter()
        config = replace(base, seed=seed)
        session = QTrainingSession(BayesianQAgent(config, TwoStepMDP.encoder()), TwoStepMDP())
        session.run(301)
        while session.done:
            session.step()
        saved_step = session.agent.interactions
        checkpoint = session.save(output / f'seed_{seed}_mid_episode.rrtl')
        resumed = QTrainingSession.load(checkpoint, TwoStepMDP())
        remaining = 600 - saved_step
        session.run(remaining)
        resumed.run(remaining)
        a, b = session.agent, resumed.agent
        assert session.history == resumed.history
        assert session.environment.snapshot() == resumed.environment.snapshot()
        for name in ('environment_rng', 'exploration_rng', 'replay_rng'):
            assert getattr(a, name).bit_generator.state == getattr(b, name).bit_generator.state
        assert [r.bit_generator.state for r in a.mcmc_rngs] == [r.bit_generator.state for r in b.mcmc_rngs]
        for p, q in zip(a.model.posteriors, b.model.posteriors):
            assert p.draws == q.draws and p.trace == q.trace
        for p, q in zip(a.last_fit.datasets, b.last_fit.datasets):
            assert p.transition_ids == q.transition_ids and p.data_hash == q.data_hash
            np.testing.assert_array_equal(p.targets, q.targets)
        batch = a.encoder.transform([TwoStepMDP.state(0), TwoStepMDP.state(1)])
        learned, optimal = a.model.predict(batch), TwoStepMDP.optimal_q(config.gamma)
        error = float(np.max(np.abs(learned - optimal)))
        assert error < 0.12
        assert learned.argmax(axis=1).tolist() == [0, 0]
        assert all(row['return'] == 1.0 for row in a.evaluate(TwoStepMDP, config.validation_seeds))
        session.save(output / f'seed_{seed}_final.rrtl')
        results.append({'seed': seed, 'interactions': a.interactions, 'fits': a.model.model_id,
                        'checkpoint_step': saved_step, 'resume_exact': True,
                        'q': learned.tolist(), 'optimal_q': optimal.tolist(), 'max_abs_q_error': error,
                        'config_hash': config.config_hash, 'seconds': time.perf_counter() - started})
    report = {'milestone': 'H5', 'environment': 'two_step_v1', 'seeds': results,
              'criteria': {'max_abs_q_error_below': 0.12, 'optimal_actions': [0, 0],
                           'resume': 'identical transitions, RNG, targets and posterior draws'},
              'limitations': ['Controlled MDP only; Atari pilot belongs to H6.',
                              'No claim of Q-learning convergence or general MCMC convergence.',
                              'Exact resume is checked on the same software/runtime.']}
    (output / 'report.json').write_text(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(validate(args.output), indent=2))
