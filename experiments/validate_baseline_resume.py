"""Gate C: baseline checkpoint continuation locally and in a fresh process."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from bayesian_rrtl.atari import AtariConfig
from bayesian_rrtl.baseline_adapter import BaselineLearnerConfig, RelationalBaselineAgent, normalize_facts
from bayesian_rrtl.baseline_checkpoint import evaluate_baseline_checkpoint
from bayesian_rrtl.baseline_training import BaselineTrainingSession
from bayesian_rrtl.comparison import make_comparison_session
from bayesian_rrtl.encoding import RelationKey, RelationalEncoder
from preprocessing import Fact


class SplitSignalEnvironment:
    """Single action, deterministic separable TD signal; no Atari or reward shaping."""
    actions = ('NOOP',)

    def __init__(self):
        self.index, self.done, self.seed = 0, True, None

    def encoder(self):
        return RelationalEncoder([RelationKey('comparative', axis, 'player_t', 'ball_t') for axis in ('x', 'y')])

    def state(self):
        return tuple(Fact('comparative', ('less', 'more')[self.index // period % 2], axis, 'player_t', 'ball_t')
                     for period, axis in ((1, 'x'), (2, 'y')))

    def reset(self, *, seed):
        self.index, self.seed, self.done = 0, seed, False
        return self.state()

    def step(self, action):
        assert action == 0 and not self.done
        reward = (-4 if self.index % 2 == 0 else 4) + (-1 if self.index // 2 % 2 == 0 else 1)
        self.index += 1
        return self.state(), float(reward), float(reward), False, False

    def runtime_manifest(self):
        return {'environment': 'split_signal_v1', 'actions': list(self.actions), 'encoder': self.encoder().to_dict()}

    def snapshot(self):
        return {'kind': 'split_signal_snapshot_v1', 'done': self.done, 'index': self.index, 'seed': self.seed}

    def validate_snapshot(self, snapshot):
        if (set(snapshot) != {'kind', 'done', 'index', 'seed'} or snapshot['kind'] != 'split_signal_snapshot_v1'
                or type(snapshot['index']) is not int or snapshot['index'] < 0):
            raise ValueError('Invalid synthetic snapshot')

    def restore(self, snapshot):
        self.validate_snapshot(snapshot)
        self.index, self.done, self.seed = snapshot['index'], snapshot['done'], snapshot['seed']

    def close(self):
        pass


def tree_data(node):
    return {'q': node.q_value, 'literal': node.literal, 'refinements': node.refinements,
            'statistics': [(lit, stats) for lit, stats in node.refinements_stats.items()],
            'depth': node.depth, 'children': [tree_data(child) for child in node.children]}


def digest(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, allow_nan=False).encode()).hexdigest()


def fingerprint(session):
    agent = session.agent
    facts = normalize_facts(session.state) if session.state is not None else ()
    return {'trees': [tree_data(root) for root in agent.root_nodes.values()],
            'q': [root.predict(facts) for root in agent.root_nodes.values()],
            'facts': [list(fact) for fact in facts], 'splits': agent.n_splits, 'interactions': session.interactions,
            'query_counts': agent.query_counts, 'learner_rng': agent.policy_rng.bit_generator.state,
            'environment_rng': session.environment_rng.bit_generator.state,
            'exploration_rng': session.exploration_rng.bit_generator.state,
            'training_seeds': session.training_seeds, 'history': session.history,
            'episodes': session.episode_rows(), 'epsilon': session.epsilon}


def continuation(session, steps):
    hashes = []
    for _ in range(steps):
        session.step()
        hashes.append(digest(fingerprint(session)))
    return {'step_hashes': hashes, 'final': fingerprint(session)}


def load_case(path, synthetic):
    return BaselineTrainingSession.load(path, environment_factory=SplitSignalEnvironment if synthetic else None)


def validate_case(backend, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    synthetic = backend == 'synthetic'
    if synthetic:
        environment = SplitSignalEnvironment()
        session = BaselineTrainingSession(RelationalBaselineAgent(environment.actions, environment.encoder(),
            BaselineLearnerConfig(eta_q=1., gamma=0., min_sample_size=6, significance_level=.05)), environment,
            environment_factory=SplitSignalEnvironment)
        pause = 12
    else:
        config = AtariConfig(game='Pong', extractor=backend, include_incomplete_states=backend == 'ocatari_ram_v1',
                             max_episode_steps=220)
        session = make_comparison_session('corrected_common', config, BaselineLearnerConfig(seed=61))
        pause = 100
    try:
        session.run(pause)
        assert not session.done
        checkpoint = session.save(output / 'progress.rrtlc')
        initial_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        initial = digest(fingerprint(session))
        splits_before = session.agent.n_splits
        if synthetic:
            assert splits_before == 1
        expected = continuation(session, 128)
        # JSON is the comparison boundary across process/type identities.
        expected = json.loads(json.dumps(expected))
        resumed = load_case(checkpoint, synthetic)
        try:
            assert digest(fingerprint(resumed)) == initial
            local = json.loads(json.dumps(continuation(resumed, 128)))
        finally:
            resumed.environment.close()
        command = [sys.executable, '-m', 'experiments.validate_baseline_resume', '--worker', str(checkpoint),
                   '--output', str(output / 'child.json')]
        if synthetic:
            command.append('--synthetic')
        subprocess.run(command, check=True, timeout=60, capture_output=True, text=True)
        child = json.loads((output / 'child.json').read_text())
        assert expected == local == child
        if synthetic:
            assert expected['final']['splits'] == 3
        final_checkpoint = session.save(output / 'final.rrtlc')
        final_sha = hashlib.sha256(final_checkpoint.read_bytes()).hexdigest()
        # Final evaluation uses the saved artifact, not the live training model.
        report = evaluate_baseline_checkpoint(final_checkpoint, (100001,), horizon=16,
            environment_factory=SplitSignalEnvironment if synthetic else None)
        assert hashlib.sha256(final_checkpoint.read_bytes()).hexdigest() == final_sha
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == initial_hash
        evidence = {'status': 'passed', 'backend': backend, 'pause_interactions': pause,
            'continuation_decisions': 128, 'mid_episode_pause': True, 'local_and_process_exact': True,
            'compared': ['actions', 'facts', 'reward/raw', 'flags', 'Q', 'statistics', 'splits',
                         'RNG', 'seeds', 'history', 'partial_episode', 'epsilon'],
            'initial_fingerprint': initial, 'continuation_sha256': digest(expected),
            'splits_before': splits_before, 'splits_after': session.agent.n_splits,
            'episode_seeds': session.training_seeds, 'learner_config': asdict(session.agent.config),
            'progress_checkpoint_sha256': initial_hash, 'final_checkpoint_sha256': final_sha,
            'evaluation': report, 'evaluation_checkpoint_unchanged': True, 'worker_command': command}
        (output / 'report.json').write_text(json.dumps(evidence, indent=2) + '\n')
        return evidence
    finally:
        session.environment.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worker', type=Path)
    parser.add_argument('--synthetic', action='store_true')
    args = parser.parse_args()
    if args.worker:
        session = load_case(args.worker, args.synthetic)
        try:
            args.output.write_text(json.dumps(continuation(session, 128)))
        finally:
            session.environment.close()
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        for backend in ('synthetic', 'legacy_visual_pong_v1', 'ocatari_ram_v1'):
            report = validate_case(backend, args.output / backend)
            print(f"{backend}: 128 exact continuation steps, splits {report['splits_before']} -> "
                  f"{report['splits_after']}", flush=True)


if __name__ == '__main__':
    main()
