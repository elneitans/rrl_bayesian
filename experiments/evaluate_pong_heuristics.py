"""New-seed, frozen Pong heuristic controls; no pilot checkpoint or test seed use."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np

from bayesian_rrtl.atari import AtariConfig
from bayesian_rrtl.evaluation import paired_bootstrap
from bayesian_rrtl.objects import make_extractor
from experiments.pong_analysis_common import ROOT, config, create_manifest, load_manifest, read, source_path, write_new

POLICIES = ('random', 'noop', 'object_reactive', 'fact_reactive')


def object_frame(environment, extractor=None):
    if environment.backend is not None:
        return environment.backend.current_frame
    if extractor is None:
        raise ValueError('Visual object probe needs a separate extractor')
    return extractor.extract(environment.env.unwrapped.ale.getScreenRGB())


def probe_actions(factory, seed, *, repetitions=8):
    """Use independent seeded episodes; measure actual player center displacement."""
    result = {}
    for action in (2, 3):  # hypotheses, not assumed direction
        env = factory()
        extractor = make_extractor(env.config.extractor) if env.backend is None else None
        try:
            env.reset(seed=seed)
            initial = None
            last = None
            for _ in range(repetitions):
                frame = object_frame(env, extractor)
                player = next(o for o in frame.objects if o.identity == 'player')
                if player.present:
                    initial = player.center[1] if initial is None else initial
                    last = player.center[1]
                _, _, _, terminated, truncated = env.step(action)
                if terminated or truncated:
                    break
            frame = object_frame(env, extractor)
            player = next(o for o in frame.objects if o.identity == 'player')
            if player.present:
                last = player.center[1]
            if initial is None or last is None:
                raise ValueError('Player was absent throughout action probe')
            result[env.actions[action]] = float(last - initial)
        finally:
            env.close()
    names = list(result)
    if not (result[names[0]] * result[names[1]] < 0 and
            min(abs(result[names[0]]), abs(result[names[1]])) >= 1):
        raise ValueError(f'No empirically opposed paddle actions: {result}')
    return {'displacements_y_pixels': result,
            'up': min(result, key=result.get), 'down': max(result, key=result.get),
            'coordinate_system': 'screen y increases downward', 'seed': seed}


def _fact_value(facts, type_, name, obj1, obj2=None):
    for fact in facts:
        if (fact.type, fact.name, fact.obj1, fact.obj2) == (type_, name, obj1, obj2):
            return str(fact.value)
    return None


class ReactivePolicy:
    """Same deadband and FIRE rule for object and Fact representations."""
    def __init__(self, up, down, actions):
        self.up, self.down = actions.index(up), actions.index(down)
        self.noop, self.fire = actions.index('NOOP'), actions.index('FIRE')
        self.missing = self.stationary = 0
        self.previous_ball = None

    def decide(self, relation, *, present, stationary):
        if not present:
            self.missing += 1
            self.stationary = 0
            self.previous_ball = None
            return self.fire if self.missing % 4 == 1 else self.noop
        self.missing = 0
        self.stationary = self.stationary + 1 if stationary else 0
        if self.stationary >= 8 and self.stationary % 8 == 0:
            return self.fire
        return {'above': self.down, 'below': self.up}.get(relation, self.noop)

    def object_action(self, frame):
        objects = {obj.identity: obj for obj in frame.objects}
        player, ball = objects['player'], objects['ball']
        if not (player.present and ball.present):
            return self.decide(None, present=False, stationary=False)
        center = ball.center
        stationary = self.previous_ball == center
        self.previous_ball = center
        delta = ball.center[1] - player.center[1]
        relation = 'above' if delta > 4 else 'below' if delta < -4 else 'same'
        return self.decide(relation, present=True, stationary=stationary)

    def fact_action(self, facts):
        # This method has no environment, backend, frame or bbox argument.
        player = _fact_value(facts, 'logical', 'present', 'player_t') == 'True'
        ball = _fact_value(facts, 'logical', 'present', 'ball_t') == 'True'
        if not (player and ball):
            return self.decide(None, present=False, stationary=False)
        relative = _fact_value(facts, 'comparative', 'y', 'player_t', 'ball_t')
        if relative is None:
            raise ValueError('Present player/ball must have a published vertical Fact')
        dy = _fact_value(facts, 'comparative', 'y', 'ball_t', 'ball_t-1')
        dx = _fact_value(facts, 'comparative', 'x', 'ball_t', 'ball_t-1')
        return self.decide({'less': 'above', 'more': 'below', 'same': 'same'}[relative],
                           present=True, stationary=(dy == dx == 'same'))


def play(factory, pipeline, policy, seed, horizon, seconds, directions):
    env = factory()
    rng = np.random.default_rng(seed + 1000 * POLICIES.index(policy))
    reactive = ReactivePolicy(directions['up'], directions['down'], env.actions)
    counts = Counter()
    unique = set()
    started = time.monotonic()
    raw_return = won = lost = 0.
    try:
        state = env.reset(seed=seed)
        for step in range(1, horizon + 1):
            if time.monotonic() - started >= seconds:
                return {'status': 'partial', 'reason': 'time_budget', 'seed': seed, 'policy': policy,
                        'steps': step - 1}
            batch = env.encoder().transform([state])
            unique.add(batch.values.tobytes() + batch.observed.tobytes())
            if policy == 'random':
                action = int(rng.integers(len(env.actions)))
            elif policy == 'noop':
                action = env.actions.index('NOOP')
            elif policy == 'object_reactive':
                if pipeline != 'ocatari':
                    raise ValueError('Object reactive control is OCAtari-only')
                action = reactive.object_action(object_frame(env))
            else:
                action = reactive.fact_action(state)
            counts[env.actions[action]] += 1
            state, _, reward, terminated, truncated = env.step(action)
            raw_return += reward
            won += max(reward, 0.)
            lost += max(-reward, 0.)
            if terminated or truncated:
                break
        truncated = bool(truncated or (step == horizon and not terminated))
        return {'status': 'completed', 'policy': policy, 'pipeline': pipeline, 'seed': seed,
            'steps': step, 'terminated': terminated, 'truncated': truncated,
            'return_scope': 'complete_game' if terminated else 'horizon_limited',
            'raw_return': raw_return, 'points_won': won, 'points_lost': lost,
            'actions': dict(counts), 'unique_encoded_states': len(unique),
            'seconds': time.monotonic() - started}
    finally:
        env.close()


def report(output, pipeline, seeds):
    available = POLICIES if pipeline == 'ocatari' else ('random', 'noop', 'fact_reactive')
    rows = {policy: {} for policy in available}
    for policy in available:
        for seed in seeds:
            path = output / pipeline / policy / f'{seed}.json'
            if path.exists():
                row = read(path)
                if row.get('seed') != seed or row.get('policy') != policy:
                    raise ValueError('Heuristic result identity mismatch')
                rows[policy][seed] = row
    paired = {}
    for policy in available:
        if policy == 'random':
            continue
        comparable = sorted(seed for seed in rows[policy] if seed in rows['random'] and
                            rows[policy][seed]['status'] == rows['random'][seed]['status'] == 'completed')
        if comparable:
            paired[policy] = paired_bootstrap({s: rows[policy][s]['raw_return'] for s in comparable},
                {s: rows['random'][s]['raw_return'] for s in comparable})
            paired[policy]['horizon_limited_seed_count'] = sum(
                rows[policy][s]['truncated'] or rows['random'][s]['truncated'] for s in comparable)
            paired[policy]['complete_game_seed_count'] = len(comparable) - paired[policy]['horizon_limited_seed_count']
    return {'pipeline': pipeline, 'policies': rows, 'paired_vs_random': paired,
            'interpretation': 'Horizon-limited returns are labelled and are not complete losses.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/pong_bart_analysis_v1.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pipeline', choices=('ocatari', 'visual'), required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--report-only', action='store_true')
    args = parser.parse_args()
    specification = config(args.config)
    output = args.output
    if not output.exists():
        create_manifest(args.config, output)
    else:
        load_manifest(output)
    seeds = [specification['smoke_seed']] if args.smoke else specification['development_seeds']
    if args.report_only:
        print(json.dumps(report(output, args.pipeline, seeds), indent=2))
        return
    from bayesian_rrtl.comparison import PongEnvironmentFactory
    from experiments.compare_pong import load_frozen
    source = load_frozen(source_path(specification['source_campaign']))
    environment = AtariConfig(**source['protocol']['environments'][args.pipeline])
    factory = PongEnvironmentFactory(environment).with_horizon(
        specification['smoke_horizon'] if args.smoke else specification['horizon'])
    probe_path = output / args.pipeline / 'action_probe.json'
    if probe_path.exists():
        directions = read(probe_path)
    else:
        directions = probe_actions(factory, specification['action_probe_seed'])
        write_new(probe_path, directions)
    policies = POLICIES if args.pipeline == 'ocatari' else ('random', 'noop', 'fact_reactive')
    for policy in policies:
        for seed in seeds:
            path = output / args.pipeline / policy / f'{seed}.json'
            if path.exists():
                continue
            row = play(factory, args.pipeline, policy, seed,
                specification['smoke_horizon'] if args.smoke else specification['horizon'],
                specification['max_seconds_per_episode'], directions)
            write_new(path, row)
            print(args.pipeline, policy, seed, row['status'], flush=True)
    result = report(output, args.pipeline, seeds)
    path = output / args.pipeline / ('heuristics_smoke.json' if args.smoke else 'heuristics.json')
    if not path.exists():
        write_new(path, result)


if __name__ == '__main__':
    main()
