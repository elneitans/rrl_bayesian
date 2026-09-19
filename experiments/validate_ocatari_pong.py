"""Steps 4/5 acceptance: real resume matrix, fixed smoke, separately budgeted MCMC.

No test split or pilot execution. Subprocess timeouts preserve completed block
checkpoints/logs; they never reduce the predeclared sampler settings.
"""
import argparse
import cProfile
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from bayesian_rrtl.agent import BayesianQAgent
from bayesian_rrtl.atari import AtariConfig, AtariRelationalEnvironment
from bayesian_rrtl.checkpoint import load_checkpoint
from bayesian_rrtl.profiling import profile_summary
from bayesian_rrtl.training import QTrainingSession
from bayesian_rrtl.q_config import BayesianQConfig
from experiments.validate_ocatari_backend import source_manifest, check_truncation, check_terminal
from experiments.validate_ocatari_resume import validate_resume
from train_atari import manifest, validate_pair

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / 'configs/bayesian_pong_ocatari_smoke.json'


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def run_budgeted(command, log, seconds):
    if not np.isfinite(seconds) or seconds <= 0:
        raise ValueError('Budget must be positive and finite')
    started = time.perf_counter()
    with log.open('w') as handle:
        try:
            result = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
                                    timeout=seconds, check=False)
            status = 'passed' if result.returncode == 0 else 'failed'
            reason = None if result.returncode == 0 else f'Worker exited with {result.returncode}'
        except subprocess.TimeoutExpired:
            status, reason = 'not_run', 'Budget exceeded; completed artifacts preserved'
    return {'status': status, 'reason': reason, 'budget_seconds': seconds,
            'seconds': time.perf_counter() - started, 'command': command, 'log': str(log)}


def train_smoke(output):
    raw = json.loads(SMOKE.read_text())
    output.mkdir(parents=True, exist_ok=False)
    env_config, config = AtariConfig(**raw['environment']), BayesianQConfig(**raw['agent'])
    env = AtariRelationalEnvironment(env_config)
    try:
        validate_pair(env_config, config, env)
        write_json(output / 'config.json', raw)
        write_json(output / 'manifest.json', manifest(env_config, config, 'bayesian', env))
        session = QTrainingSession(BayesianQAgent(config, env.encoder()), env)
        agent = session.agent
        session.save(output / 'progress_checkpoint.rrtl')
        encoder = env.encoder()
        present = {name: 0 for name in ('player', 'ball', 'enemy')}
        missing = np.zeros(len(encoder.catalog), dtype=int)
        actions = [0] * len(config.actions)
        distinct = set()
        curves = []
        profiler = cProfile.Profile()
        started = time.perf_counter()
        profiler.enable()
        with (output / 'train.csv').open('w', newline='') as handle:
            writer = None
            for _ in range(raw['interactions']):
                row = session.step()
                if writer is None:
                    writer = csv.DictWriter(handle, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                encoded = encoder.transform([session.state])
                distinct.add(encoded.values.tobytes())
                missing += (~encoded.observed[0]).astype(int)
                for obj in env.backend.current_frame.objects:
                    present[obj.identity] += int(obj.present)
                actions[row['action']] += 1
                if agent.interactions % config.fit_interval == 0:
                    handle.flush()
                    curve = {'interaction': agent.interactions, 'fits': agent.model.model_id,
                             'completed_episodes': sum(r['terminated'] or r['truncated'] for r in session.history),
                             'cumulative_raw_return': sum(r['raw_reward'] for r in session.history)}
                    curves.append(curve)
                    session.save(output / 'progress_checkpoint.rrtl')
                    write_json(output / 'progress.json', curve)
                    write_json(output / 'learning_curve.json', curves)
                    print(json.dumps(curve), flush=True)
        profiler.disable()
        seconds = time.perf_counter() - started
        checkpoint = session.save(output / 'final_checkpoint.rrtl')
        loaded, _ = load_checkpoint(checkpoint, expected_config=config)
        fits = [row['interaction'] for row in loaded.fit_history]
        finite = all(np.isfinite([row['reward'], row['raw_reward']]).all() for row in session.history)
        for dataset in loaded.last_fit.datasets:
            finite &= bool(np.isfinite(dataset.targets).all())
            if len(dataset.batch):
                finite &= bool(np.isfinite(loaded.model.predict(dataset.batch)).all())
        passed = (loaded.interactions == raw['interactions'] == 2048 and fits == list(range(256, 2049, 256))
                  and finite and all(actions) and len(distinct) > 1)
        report = {'status': 'passed' if passed else 'failed', 'interactions': agent.interactions,
            'fits': agent.model.model_id, 'fit_interactions': fits, 'finite_q_targets_rewards': bool(finite),
            'training_seconds': seconds, 'checkpoint': str(checkpoint), 'checkpoint_bytes': checkpoint.stat().st_size,
            'checkpoint_reloaded': True, 'distinct_encoded_states': len(distinct),
            'action_counts': dict(zip(config.actions, actions)), 'present_frames': present,
            'missing_fact_counts': missing.tolist(), 'catalog': encoder.to_dict(),
            'positive_points': sum(r['raw_reward'] > 0 for r in session.history),
            'negative_points': sum(r['raw_reward'] < 0 for r in session.history),
            'terminated': sum(r['terminated'] for r in session.history),
            'truncated': sum(r['truncated'] for r in session.history),
            'training_seeds': agent.training_seeds, 'profile': profile_summary(profiler),
            'adapter_profile': env.profile}
        write_json(output / 'fits.json', agent.fit_history)
        write_json(output / 'encoder.json', encoder.to_dict())
        write_json(output / 'report.json', report)
        profiler.dump_stats(str(output / 'training.prof'))
        if not passed:
            raise AssertionError('Smoke acceptance criteria failed')
        return report
    finally:
        env.close()


def validate(output, *, smoke_budget=600, diagnostics_budget=600):
    output.mkdir(parents=True, exist_ok=False)
    report = {'scope': 'ocatari_pong_steps_4_5', 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'criteria': {}, 'not_run': ['pilot_20480_per_seed', 'test_evaluation', 'efficacy_campaign']}
    report_path = output / 'validation.json'
    try:
        report['runtime'] = source_manifest()
        write_json(output / 'config.json', json.loads(SMOKE.read_text()))
        report['criteria']['truncation'] = check_truncation()
        report['criteria']['natural_terminal'] = check_terminal()
        report['criteria']['resume'] = validate_resume(output / 'resume')
        write_json(report_path, report)
        command = [sys.executable, '-m', 'experiments.validate_ocatari_pong', '--worker',
                   '--output', str(output / 'smoke')]
        smoke = run_budgeted(command, output / 'smoke.log', smoke_budget)
        report['criteria']['smoke'] = smoke
        result = output / 'smoke/report.json'
        if result.exists():
            smoke['results'] = json.loads(result.read_text())
        progress = output / 'smoke/progress.json'
        if progress.exists():
            smoke['last_completed_block'] = json.loads(progress.read_text())
        write_json(report_path, report)
        if smoke['status'] == 'passed':
            checkpoint = output / 'smoke/final_checkpoint.rrtl'
            original_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            diagnostics = json.loads(SMOKE.read_text())['diagnostics']
            command = [sys.executable, '-m', 'experiments.diagnose_atari', '--checkpoint', str(checkpoint),
                       '--output', str(output / 'diagnostics'), '--chains', str(diagnostics['chains']),
                       '--burn-in', str(diagnostics['burn_in']), '--draws', str(diagnostics['draws']),
                       '--seed', str(diagnostics['seed'])]
            result = run_budgeted(command, output / 'diagnostics.log', diagnostics_budget)
            result['checkpoint_unchanged'] = original_hash == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            if not result['checkpoint_unchanged']:
                raise AssertionError('Diagnostics modified checkpoint')
            diagnostic_report = output / 'diagnostics/report.json'
            if diagnostic_report.exists():
                result['results'] = json.loads(diagnostic_report.read_text())
            report['criteria']['diagnostics'] = result
        else:
            report['criteria']['diagnostics'] = {'status': 'not_run', 'reason': 'Smoke not completed'}
        states = [item['status'] for item in report['criteria'].values()]
        report['status'] = 'failed' if 'failed' in states else 'not_run' if 'not_run' in states else 'passed'
    except Exception as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        write_json(report_path, report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--smoke-budget-seconds', type=float, default=600)
    parser.add_argument('--diagnostics-budget-seconds', type=float, default=600)
    args = parser.parse_args()
    if args.worker:
        train_smoke(args.output.resolve())
    else:
        report = validate(args.output.resolve(), smoke_budget=args.smoke_budget_seconds,
                          diagnostics_budget=args.diagnostics_budget_seconds)
        print(json.dumps({'status': report['status'], 'output': str(args.output)}), flush=True)
        if report['status'] == 'failed':
            raise SystemExit(1)
