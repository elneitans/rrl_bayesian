"""Read-only campaign audit and bounded greedy traces of completed checkpoints.

Use the original Linux environment for traces. Never bypass checkpoint runtime
checks. JSON-only --summary-only works without Atari and includes partial runs.
Diagnostic episodes use validation seeds only, never reserved test seeds.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def audit_logs(campaign):
    campaign = Path(campaign)
    manifest = read(campaign / 'protocol.json')
    if sha(campaign / 'protocol.json') != (campaign / 'protocol.sha256').read_text().strip():
        raise ValueError('Protocol checksum mismatch')
    results = []
    for cell in manifest['cells']:
        folder = campaign / cell['id']
        if not (folder / 'status.json').exists():
            results.append({'cell': cell['id'], 'state': 'not_run'})
            continue
        status = read(folder / 'status.json')
        rows = read(folder / 'train.json')
        if len(rows) != status['interactions'] or any(r['transition_id'] != i for i, r in enumerate(rows)):
            raise ValueError(f"Inconsistent training log: {cell['id']}")
        checkpoint = folder / status['checkpoint']
        if sha(checkpoint) != status['checkpoint_sha256']:
            raise ValueError(f"Checkpoint checksum mismatch: {cell['id']}")
        reserved = set(manifest['protocol']['test_seeds'])
        if any(r['episode_seed'] in reserved for r in rows):
            raise ValueError('Reserved test seed found in training')
        actions = Counter(r['action'] for r in rows)
        evaluations = []
        for path in sorted(folder.glob('validation_*.json'), key=lambda p: int(p.stem.split('_')[-1])):
            report = read(path)['report']
            evaluations.append({'interaction': int(path.stem.split('_')[-1]),
                'mean_return': report['mean_raw_return'], 'truncation_fraction': report['truncation_fraction']})
        results.append({'cell': cell['id'], 'state': status['state'], 'decisions': len(rows),
            'episodes_started': len({r['episode_id'] for r in rows}),
            'episodes_terminated': sum(r['terminated'] for r in rows),
            'episodes_truncated_only': sum(r['truncated'] and not r['terminated'] for r in rows),
            'partial_episode': bool(rows and not (rows[-1]['terminated'] or rows[-1]['truncated'])),
            'last_training_epsilon': rows[-1]['epsilon'] if rows else None,
            'action_counts': dict(actions), 'points_won_training': sum(r['raw_reward'] > 0 for r in rows),
            'points_lost_training': sum(r['raw_reward'] < 0 for r in rows),
            'training_seconds': status['training_seconds'], 'evaluation_seconds': status['evaluation_seconds'],
            'checkpoint_sha256': status['checkpoint_sha256'], 'evaluations': evaluations})
    return manifest, results


def q_values(agent, state):
    if hasattr(agent, 'root_nodes'):
        return agent.predict_q(state)
    return agent.model.predict(agent.encode_state(state))[0]


def trace_episode(agent, environment, seed, horizon, output, *, seconds=600, gif=False):
    """Dedicated loaded agent + fresh env; predict only, no observe/fit or policy RNG mutation."""
    import numpy as np
    from bayesian_rrtl.baseline_adapter import normalize_facts
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(seed)
    state = environment.reset(seed=seed)
    actions = environment.actions
    counts, unique = Counter(), set()
    total = won = lost = switches = longest = streak = empty = 0
    previous = None
    margins, frames = [], []
    started = time.monotonic()
    stop = 'horizon'
    steps = 0
    with (output / 'trace.jsonl').open('w') as handle:
        for index in range(horizon):
            if time.monotonic() - started >= seconds:
                stop = 'time_budget'
                break
            batch = agent.encoder.transform([state])
            unique.add(batch.values.tobytes() + batch.observed.tobytes())
            empty += int(not batch.observed.any())
            q = np.asarray(q_values(agent, state), dtype=float)
            if q.shape != (len(actions),) or not np.isfinite(q).all():
                raise ValueError('Invalid Q vector')
            ties = np.flatnonzero(q == q.max())
            action = int(rng.choice(ties))  # Same epsilon=0 tie rule as common evaluation.
            margin = float(np.sort(q)[-1] - np.sort(q)[-2]) if len(q) > 1 else 0.
            margins.append(margin)
            switches += int(previous is not None and previous != action)
            streak = streak + 1 if previous == action else 1
            longest = max(longest, streak)
            previous = action
            counts[actions[action]] += 1
            facts = [dict(f._asdict()) for f in normalize_facts(state)]
            routes = None
            if hasattr(agent, 'root_nodes'):
                routes = {}
                for name, root in agent.root_nodes.items():
                    node = root.get_state_leaf(normalize_facts(state))
                    routes[name] = {'depth': node.depth, 'internal_fallback': node.literal is not None}
            row = {'step': index, 'facts': facts, 'encoded_values': batch.values[0].tolist(),
                'observed': batch.observed[0].tolist(), 'q': dict(zip(actions, q.tolist())),
                'greedy_ties': [actions[i] for i in ties], 'q_margin': margin,
                'action': actions[action], 'action_index': action, 'baseline_routes': routes}
            if gif and index % 4 == 0 and len(frames) < 300:
                from PIL import Image
                base = (environment.env.unwrapped if environment.backend is None
                        else environment.backend.env._env.unwrapped)
                frames.append(Image.fromarray(base.ale.getScreenRGB()).copy())
            state, reward, raw, terminal, truncated = environment.step(action)
            total += raw
            won += max(raw, 0)
            lost += max(-raw, 0)
            row.update(reward=reward, raw_reward=raw, terminated=terminal, truncated=truncated)
            handle.write(json.dumps(row, allow_nan=False) + '\n')
            handle.flush()  # Survives interruption during a later step.
            steps += 1
            if terminal or truncated:
                stop = 'terminated' if terminal else 'environment_truncation'
                break
    if frames:
        frames[0].save(output / 'preview.gif', save_all=True, append_images=frames[1:],
                       duration=267, loop=0)
    result = {'seed': seed, 'steps': steps, 'stop_reason': stop, 'raw_return': total,
        'points_won': won, 'points_lost': lost, 'actions': dict(counts), 'switches': switches,
        'longest_same_action_streak': longest, 'unique_encoded_states': len(unique), 'empty_states': empty,
        'mean_q_margin': float(np.mean(margins)) if margins else None,
        'seconds': time.monotonic() - started, 'gif_frames': len(frames),
        'gif_note': 'Preview only: every 4 decisions, maximum 300 frames; not necessarily a full game.'}
    write(output / 'summary.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--summary-only', action='store_true')
    parser.add_argument('--cell', help='Optional exact cell ID; otherwise all completed cells')
    parser.add_argument('--horizon', type=int, default=2000)
    parser.add_argument('--seconds-per-cell', type=float, default=600)
    parser.add_argument('--gif', action='store_true')
    args = parser.parse_args()
    import math
    if args.horizon < 1 or not math.isfinite(args.seconds_per_cell) or args.seconds_per_cell <= 0:
        parser.error('Positive finite diagnostic budgets required')
    manifest, audit = audit_logs(args.campaign)
    if args.cell and args.cell not in [c['id'] for c in manifest['cells']]:
        parser.error('Unknown cell')
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'source_protocol_sha256': sha(args.campaign / 'protocol.json'), 'training': audit,
              'traces': {}, 'mode': 'summary_only' if args.summary_only else 'greedy_diagnostic',
              'script_sha256': sha(__file__)}
    write(args.output / 'report.json', report)
    if args.summary_only:
        return 0
    from experiments.compare_pong import load_frozen, load_session
    from bayesian_rrtl.comparison import PongEnvironmentFactory
    from bayesian_rrtl.atari import AtariConfig
    load_frozen(args.campaign)  # Preserve existing source compatibility checks.
    seed = manifest['protocol']['validation_seeds'][0]
    if seed in manifest['protocol']['test_seeds']:
        raise ValueError('Refusing reserved test seed')
    failed = False
    for cell, summary in zip(manifest['cells'], audit):
        if summary['state'] != 'completed' or (args.cell and cell['id'] != args.cell):
            continue
        checkpoint = args.campaign / cell['id'] / 'final_checkpoint.rrtl'
        before = sha(checkpoint)
        session = env = None
        try:
            session = load_session(cell, checkpoint)
            env = PongEnvironmentFactory(AtariConfig(**cell['environment'])).with_horizon(args.horizon)()
            result = trace_episode(session.agent, env, seed, args.horizon, args.output / cell['id'],
                                   seconds=args.seconds_per_cell, gif=args.gif)
            report['traces'][cell['id']] = {'status': 'recorded', 'checkpoint_sha256': before, **result}
        except Exception as exc:
            failed = True
            report['traces'][cell['id']] = {'status': 'failed', 'error': f'{type(exc).__name__}: {exc}'}
        finally:
            if env is not None:
                env.close()
            if session is not None:
                session.environment.close()
            if sha(checkpoint) != before:
                raise RuntimeError('Diagnostic changed checkpoint')
        write(args.output / 'report.json', report)
        print(cell['id'], report['traces'][cell['id']]['status'], flush=True)
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
