"""Gate B: fixed-action input parity, two independent ALE instances per backend.

This is a collector/semantics check, not the four-cell learning smoke or test.
Captured inputs precede observe; BART fits are disabled by an explicit warmup
outside this bounded trace. No sampler or baseline criterion is changed.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np

from bayesian_rrtl.atari import AtariConfig
from bayesian_rrtl.baseline_adapter import BaselineLearnerConfig, normalize_facts
from bayesian_rrtl.comparison import make_comparison_session
from bayesian_rrtl.games.pong import PONG_ACTIONS
from bayesian_rrtl.q_config import BayesianQConfig


def encoded(encoder, state):
    batch = encoder.transform([state])
    return {'catalog_hash': batch.catalog_hash, 'values': batch.values.tolist(),
            'observed': batch.observed.tolist()}


def validate_inputs(backend, *, seed=61, horizon=512, interactions=1030):
    config = AtariConfig(game='Pong', extractor=backend, include_incomplete_states=backend == 'ocatari_ram_v1',
                         max_episode_steps=horizon)
    baseline_config = BaselineLearnerConfig(seed=seed)
    bart_config = BayesianQConfig(actions=PONG_ACTIONS, seed=seed, gamma=.99,
        reward_min=config.reward_bounds[0], reward_max=config.reward_bounds[1], epsilon_decay_steps=50000,
        warmup_interactions=interactions + 1)
    sessions = []
    records = [[], []]
    captures = [[], []]
    calls = [0, 0]
    started = time.perf_counter()
    try:
        sessions.append(make_comparison_session('corrected_common', config, baseline_config))
        sessions.append(make_comparison_session('bart', config, bart_config))
        assert sessions[0].environment is not sessions[1].environment
        assert sessions[0].environment.config.resolved_dict() == sessions[1].environment.config.resolved_dict()
        def observe_capture(index):
            agent = sessions[index].agent
            original = agent.observe
            def observe(state, action, reward, following, terminated, truncated, **kwargs):
                # Baseline's boundary normalizes only logical values; compare the
                # canonical meaning against BART's actual encoded input.
                effective = normalize_facts(state) if index == 0 else state
                captures[index].append({'state': encoded(agent.encoder, effective),
                    'next_state': encoded(agent.encoder, following), 'action': action, 'reward': reward,
                    'terminated': terminated, 'truncated': truncated})
                return original(state, action, reward, following, terminated, truncated, **kwargs)
            return observe
        def counted_step(index):
            original = sessions[index].environment.step
            def step(action):
                calls[index] += 1
                return original(action)
            return step
        # Both loops execute their normal observe path; only action selection is
        # scripted, since learned Q functions are expected to diverge online.
        from contextlib import ExitStack
        with ExitStack() as stack:
            for index, session in enumerate(sessions):
                actions = iter(i % len(PONG_ACTIONS) for i in range(interactions))
                stack.enter_context(patch.object(session.agent, 'select_action',
                    lambda *args, _actions=actions, **kwargs: next(_actions)))
                stack.enter_context(patch.object(session.agent, 'observe', observe_capture(index)))
                stack.enter_context(patch.object(session.environment, 'step', counted_step(index)))
            for _ in range(interactions):
                for index, session in enumerate(sessions):
                    row = session.step()
                    records[index].append({key: value for key, value in row.items() if key != 'learner_metrics'})
                assert captures[0][-1] == captures[1][-1], 'Learner inputs diverged'
                assert records[0][-1] == records[1][-1], 'Collector logs diverged'
        rows = records[0]
        points = sum(row['raw_reward'] != 0 and not row['terminated'] for row in rows)
        truncations = sum(row['truncated'] for row in rows)
        resets = len(sessions[0].training_seeds)
        assert points > 0 and truncations >= 2 and resets >= 3
        assert calls == [interactions, interactions]
        assert sessions[0].agent.query_counts['update']['queries'] == interactions
        assert len(sessions[1].agent.replay) == interactions
        assert sessions[1].agent.model.model_id == 0
        for record in rows:
            expected = float(np.sign(record['raw_reward'])) + (.1 if backend == 'legacy_visual_pong_v1' else 0.)
            assert record['reward'] == expected, 'Reward transformed more than once or incorrectly'
        return {'status': 'passed', 'backend': backend, 'seconds': time.perf_counter() - started,
                'environment': config.resolved_dict(), 'training_seed': seed,
                'episode_seeds': sessions[0].training_seeds,
                'policy': asdict(sessions[0].policy_config), 'fixed_actions': 'decision_index % 6',
                'interactions_per_learner': interactions, 'environment_step_calls': calls,
                'nonterminal_points': points, 'truncations': truncations, 'resets': resets,
                'last_episode_scope': rows[-1]['return_scope'],
                'baseline_splits': sessions[0].agent.n_splits, 'bart_fits': 0,
                'bart_warmup_interactions': bart_config.warmup_interactions,
                'captured_before_observe': True, 'semantic_inputs_equal': True,
                'reward_raw_flags_logs_equal': True, 'single_reward_transform': True,
                'input_trace_sha256': hashlib.sha256(json.dumps(captures[0], sort_keys=True).encode()).hexdigest(),
                'common_log_sha256': hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest(),
                'runtime_environment': sessions[0].environment.runtime_manifest()}
    finally:
        for session in sessions:
            session.environment.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    for backend in ('legacy_visual_pong_v1', 'ocatari_ram_v1'):
        result = validate_inputs(backend)
        (args.output / f'{backend}.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        print(f"{backend}: {result['interactions_per_learner']} paired transitions, "
              f"{result['nonterminal_points']} points, {result['truncations']} truncations", flush=True)


if __name__ == '__main__':
    main()
