"""Strict cross-process Pong snapshot acceptance; no training/test evaluation."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import pickle
import subprocess
import sys
import time

import numpy as np

from bayesian_rrtl.atari import AtariConfig, AtariRelationalEnvironment


def json_data(value):
    if isinstance(value, dict):
        return {k: json_data(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_data(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def observation(env, result):
    facts, reward, raw, terminal, truncated = result
    return json_data({'objects': asdict(env.backend.current_frame),
                      'detector': env.backend.detector_state(),
                      'facts': sorted([list(f) for f in facts], key=repr),
                      'reward': reward, 'raw_reward': raw,
                      'terminated': terminal, 'truncated': truncated})


def continue_case(case):
    env = AtariRelationalEnvironment(AtariConfig(**case['snapshot']['config']))
    try:
        env.restore(case['snapshot'])
        rows = []
        for action in case['actions']:
            result = env.step(action)
            rows.append(observation(env, result))
            if result[-2] or result[-1]:
                break
        return rows
    finally:
        env.close()


def collect_cases():
    config = AtariConfig(game='Pong', extractor='ocatari_ram_v1', include_incomplete_states=True,
                         max_episode_steps=27000)
    env = AtariRelationalEnvironment(config)
    cases, rows = {}, []
    actions = np.random.default_rng(41).integers(0, 6, size=27000).tolist()
    names = {'mid_rally', 'before_point', 'after_point', 'before_absence', 'absent',
             'before_reappearance', 'reappeared'}
    try:
        env.reset(seed=41)
        seen_ball = False
        for index, action in enumerate(actions):
            before = env.snapshot()
            was_present = next(o.present for o in env.backend.current_frame.objects if o.identity == 'ball')
            result = env.step(action)
            rows.append(observation(env, result))
            now_present = next(o.present for o in env.backend.current_frame.objects if o.identity == 'ball')
            after = env.snapshot()
            candidates = {}
            if index >= 40 and now_present and result[2] == 0:
                candidates['mid_rally'] = (index + 1, after)
            if result[2] != 0:
                candidates.update(before_point=(index, before), after_point=(index + 1, after))
            if was_present and not now_present:
                candidates.update(before_absence=(index, before), absent=(index + 1, after))
            if seen_ball and not was_present and now_present:
                candidates.update(before_reappearance=(index, before), reappeared=(index + 1, after))
            seen_ball |= now_present
            for name, (position, snapshot) in candidates.items():
                if name not in cases:
                    cases[name] = {'position': position, 'snapshot': snapshot}
            if names <= cases.keys() and len(rows) >= max(c['position'] for c in cases.values()) + 128:
                break
            if result[-2] or result[-1]:
                raise AssertionError('Episode ended before the boundary acceptance matrix was collected')
        if not names <= cases.keys():
            raise AssertionError(f'Missing real boundary cases: {names - cases.keys()}')
        for case in cases.values():
            position = case['position']
            case['actions'] = actions[position:position + 128]
            case['expected'] = rows[position:position + 128]
        return cases
    finally:
        env.close()


def validate_resume(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    cases = collect_cases()
    config = AtariConfig(game='Pong', extractor='ocatari_ram_v1', include_incomplete_states=True,
                         max_episode_steps=42)
    env = AtariRelationalEnvironment(config)
    try:
        env.reset(seed=41)
        for i in range(41):
            env.step(i % 6)
        snapshot = env.snapshot()
        rows = [observation(env, env.step(0))]
        cases['before_truncation'] = {'position': 41, 'snapshot': snapshot,
                                      'actions': [0] * 128, 'expected': rows}
    finally:
        env.close()
    # Payloads contain explicit state data, never a live environment or handles.
    path = output / 'cases.pkl'
    path.write_bytes(pickle.dumps(cases, protocol=pickle.HIGHEST_PROTOCOL))
    local = {name: continue_case(case) for name, case in cases.items()}
    command = [sys.executable, '-m', 'experiments.validate_ocatari_resume', '--worker', str(path),
               '--output', str(output / 'child.json')]
    subprocess.run(command, check=True)
    child = json.loads((output / 'child.json').read_text())
    report = {'status': 'passed', 'worker_command': command, 'cases': {}}
    for name, case in cases.items():
        expected = case['expected']
        if local[name] != expected or child[name] != expected:
            raise AssertionError(f'Future trajectory mismatch for {name}')
        report['cases'][name] = {'status': 'passed', 'snapshot_after_decisions': case['position'],
            'compared_steps': len(expected), 'objects_facts_flags_rewards_exact': True,
            'detector_temporal_fields_exact': True, 'max_deviation': 0,
            'trajectory_sha256': hashlib.sha256(json.dumps(expected, sort_keys=True).encode()).hexdigest()}
    report['seconds'] = time.perf_counter() - started
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worker', type=Path)
    args = parser.parse_args()
    if args.worker:
        cases = pickle.loads(args.worker.read_bytes())  # trusted local acceptance artifacts only
        args.output.write_text(json.dumps({name: continue_case(case) for name, case in cases.items()}))
    else:
        print(json.dumps(validate_resume(args.output), indent=2))
