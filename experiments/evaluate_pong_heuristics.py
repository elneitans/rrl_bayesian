"""New-seed, frozen Pong heuristic controls; no pilot checkpoint or test seed use."""
import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from bayesian_rrtl.atari import AtariConfig
from bayesian_rrtl.evaluation import paired_bootstrap
from bayesian_rrtl.objects import make_extractor
from experiments.pong_analysis_common import ROOT, create_manifest, load_manifest, pilot_protocol

from experiments.pong_analysis_runtime import (WorkUnit, atomic_json, digest, episode,
    extension_arguments, identity, mode_root, unit_lock)

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


@dataclass(frozen=True)
class Observation:
    relation: str | None
    stationary: bool
    cause: str


class ObjectAdapter:
    def __init__(self):
        self.previous = None

    def reset(self):
        self.previous = None

    def observe(self, frame):
        objects = {obj.identity: obj for obj in frame.objects}
        player, ball = objects['player'], objects['ball']
        if not (player.present and ball.present):
            self.previous = None
            return Observation(None, False, 'observed_absence')
        stationary = self.previous == ball.center
        had_history = self.previous is not None
        self.previous = ball.center
        delta = ball.center[1]-player.center[1]
        relation = 'above' if delta > 4 else 'below' if delta < -4 else 'same'
        return Observation(relation, stationary, 'available' if had_history else 'insufficient_history')


class FactAdapter:
    """Only Facts cross this boundary; visual availability is not physical presence."""
    def __init__(self, convention='ocatari'):
        if convention not in ('ocatari', 'visual', 'common'):
            raise ValueError('Unknown coordinate convention')
        self.convention = convention

    def observe(self, facts):
        if not facts:
            return Observation(None, False, 'empty_state')
        if self.convention != 'visual':
            presence = [_fact_value(facts, 'logical', 'present', name+'_t') for name in ('player', 'ball')]
            if 'False' in presence:
                return Observation(None, False, 'observed_absence')
            if None in presence:
                return Observation(None, False, 'relation_unavailable')
        relative = _fact_value(facts, 'comparative', 'y', 'player_t', 'ball_t')
        if relative is None:
            return Observation(None, False, 'relation_unavailable')
        mapping = {'less': 'above', 'more': 'below', 'same': 'same'}
        if self.convention == 'visual':
            mapping = {'more': 'above', 'less': 'below', 'same': 'same'}
        dy = _fact_value(facts, 'comparative', 'y', 'ball_t', 'ball_t-1')
        dx = _fact_value(facts, 'comparative', 'x', 'ball_t', 'ball_t-1')
        return Observation(mapping[relative], dy == dx == 'same',
                           'available' if dx is not None and dy is not None else 'insufficient_history')


class ReactivePolicy:
    """Control independent of coordinates; adapters supply comparable observations."""
    def __init__(self, up, down, actions, convention='ocatari'):
        self.up, self.down = actions.index(up), actions.index(down)
        self.noop, self.fire = actions.index('NOOP'), actions.index('FIRE')
        self.object_adapter = ObjectAdapter()
        self.fact_adapter = FactAdapter(convention)
        self.reset()

    def reset(self):
        self.missing = self.stationary = 0
        self.object_adapter.reset()
        self.last_reason = 'reset'
        self.last_cause = 'insufficient_history'

    def decide(self, observation):
        self.last_cause = observation.cause
        if observation.relation is None:
            self.missing += 1
            self.stationary = 0
            action = self.fire if self.missing % 4 == 1 else self.noop
            self.last_reason = 'recovery_fire' if action == self.fire else 'insufficient_observation'
            return action
        self.missing = 0
        self.stationary = self.stationary+1 if observation.stationary else 0
        if self.stationary >= 8 and self.stationary % 8 == 0:
            self.last_reason = 'recovery_fire'
            return self.fire
        self.last_reason = 'alignment' if observation.relation == 'same' else 'movement'
        return {'above': self.down, 'below': self.up}.get(observation.relation, self.noop)

    def object_action(self, frame):
        return self.decide(self.object_adapter.observe(frame))

    def fact_action(self, facts):
        return self.decide(self.fact_adapter.observe(facts))


class HeuristicController:
    def __init__(self, policy, pipeline, directions):
        self.policy, self.pipeline, self.directions = policy, pipeline, directions
        self.reactive = None
        self.extractor = None
        self.log = []

    def action(self, state, env, rng):
        if self.reactive is None:
            self.reactive = ReactivePolicy(self.directions['up'], self.directions['down'],
                                           env.actions, self.pipeline)
        if self.policy == 'random':
            action, reason, cause = int(rng.integers(len(env.actions))), 'random', 'not_applicable'
        elif self.policy == 'noop':
            action, reason, cause = env.actions.index('NOOP'), 'noop', 'not_applicable'
        else:
            if self.policy == 'object_reactive':
                if env.backend is None and self.extractor is None:
                    self.extractor = make_extractor(env.config.extractor)
                action = self.reactive.object_action(object_frame(env, self.extractor))
            else:
                action = self.reactive.fact_action(state)
            reason, cause = self.reactive.last_reason, self.reactive.last_cause
        self.log.append({'action': action, 'reason': reason, 'cause': cause})
        return action

    def after_step(self, reward):
        if reward and self.reactive is not None:
            self.reactive.reset()


def play(factory, pipeline, policy, seed, horizon, seconds, directions, *, folder=None,
         expected=None, extend_seconds=0., extension_id=None):
    # Compatibility for short in-memory callers; persisted runners always provide a unit.
    import tempfile
    if folder is None:
        with tempfile.TemporaryDirectory() as temporary:
            return play(factory, pipeline, policy, seed, horizon, seconds, directions,
                        folder=Path(temporary), expected={'seed': seed, 'horizon': horizon})
    row = episode(factory, HeuristicController(policy, pipeline, directions), folder, expected,
                  seed=seed, horizon=horizon, seconds=seconds,
                  rng_seed=seed+1000*POLICIES.index(policy), extend_seconds=extend_seconds,
                  extension_id=extension_id)
    return {**row, 'policy': policy, 'pipeline': pipeline}


def episode_identity(manifest, pipeline, policy, seed, smoke, directions):
    cfg = manifest['config']
    common = pipeline.startswith('common_')
    actual_pipeline = pipeline.removeprefix('common_')
    protocol = pilot_protocol(manifest)
    from bayesian_rrtl.games.pong import PongRelationsV1
    from bayesian_rrtl.objects import MultiGameRelations
    encoder = (PongRelationsV1() if common or pipeline == 'ocatari' else MultiGameRelations('Pong')).encoder()
    horizon = cfg['smoke_horizon'] if smoke else cfg['horizon']
    return identity(manifest, pipeline=pipeline, stage='heuristics', method=policy,
        dataset='new_development_episodes', variant='heuristic_v2', smoke=smoke, seeds=[seed],
        effective={'environment': cfg['common_protocol'] if common else protocol['environments'][actual_pipeline], 'directions': directions},
        hashes={'catalog': encoder.catalog_hash, 'data': None, 'targets': None},
        evaluation={'model': digest(directions), 'policy': policy, 'horizon': horizon})


def report(output, pipeline, seeds, *, manifest=None, smoke=False, directions=None):
    rows = {policy: {} for policy in POLICIES}
    for policy in POLICIES:
        for seed in seeds:
            path = output / policy / str(seed)
            if (path/'state.json').exists():
                expected = episode_identity(manifest, pipeline, policy, seed, smoke, directions)
                with unit_lock(path):
                    saved = WorkUnit(path, expected).load()
                rows[policy][seed] = {**saved['result'], 'policy': policy, 'pipeline': pipeline}
    paired = {}
    for policy in POLICIES[1:]:
        comparable = sorted(seed for seed in rows[policy] if seed in rows['random'] and
                            rows[policy][seed]['status'] == rows['random'][seed]['status'] == 'completed')
        if comparable:
            paired[policy] = paired_bootstrap({s: rows[policy][s]['raw_return'] for s in comparable},
                {s: rows['random'][s]['raw_return'] for s in comparable})
            paired[policy]['paired_seeds'] = comparable
            paired[policy]['horizon_limited_seed_count'] = sum(
                rows[policy][s]['truncated'] or rows['random'][s]['truncated'] for s in comparable)
            paired[policy]['complete_game_seed_count'] = len(comparable)-paired[policy]['horizon_limited_seed_count']
            paired[policy]['by_scope'] = {}
            for scope in ('complete_game', 'horizon_limited'):
                same_scope = [s for s in comparable if rows[policy][s]['return_scope'] ==
                              rows['random'][s]['return_scope'] == scope]
                if same_scope:
                    paired[policy]['by_scope'][scope] = paired_bootstrap(
                        {s: rows[policy][s]['raw_return'] for s in same_scope},
                        {s: rows['random'][s]['raw_return'] for s in same_scope})
            paired[policy]['mixed_scope_seed_count'] = sum(
                rows[policy][s]['return_scope'] != rows['random'][s]['return_scope'] for s in comparable)
    summaries = {}
    for policy in POLICIES:
        summaries[policy] = {}
        for scope in ('complete_game', 'horizon_limited', 'administrative_partial'):
            group = [r for r in rows[policy].values() if r['return_scope'] == scope]
            if group:
                values = np.array([r['raw_return'] for r in group])
                summaries[policy][scope] = {'episodes': len(group), 'median_raw_return': float(np.median(values)),
                    'standard_deviation_raw_return': float(values.std()),
                    'iqr_raw_return': float(np.quantile(values, .75)-np.quantile(values, .25)),
                    'points_won': sum(r['points_won'] for r in group),
                    'points_lost': sum(r['points_lost'] for r in group)}
    all_done = all(len(rows[p]) == len(seeds) and all(r['status']=='completed' for r in rows[p].values()) for p in POLICIES)
    return {'pipeline': pipeline, 'status': 'completed' if all_done else 'partial',
            'policies': rows, 'paired_vs_random': paired, 'summaries_by_scope': summaries,
            'interpretation': 'Horizon-limited returns are labelled; controller failure is not proof of representation failure.'}


def assisted_gate(result, seeds):
    paired = result.get('paired_vs_random', {}).get('object_reactive', {})
    rows = result.get('policies', {}).get('object_reactive', {})
    available = all((rows.get(s) or rows.get(str(s), {})).get('status') == 'completed' for s in seeds)
    positive = sum((rows.get(s) or rows.get(str(s), {})).get('points_won', 0) for s in seeds) > 0
    paired_valid = sorted(paired.get('paired_seeds', [])) == sorted(seeds)
    enabled = available and paired_valid and positive and paired.get('mean_difference', 0) > 0
    return {'enabled': enabled, 'required_results_available': available,
            'valid_paired_random': paired_valid, 'positive_points': positive,
            'interpretation': 'Operational gate, not conclusive superiority evidence.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/pong_bart_analysis_v2.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pipeline', choices=('ocatari', 'visual'), required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--report-only', action='store_true')
    parser.add_argument('--common', action='store_true', help='New common contract, separate identities and output')
    extension_arguments(parser)
    args = parser.parse_args()
    manifest = create_manifest(args.config, args.output)
    load_manifest(args.output, writable=True)
    cfg = manifest['config']
    seeds = [cfg['smoke_seed']] if args.smoke else cfg['development_seeds']
    output = mode_root(args.output, args.pipeline, 'heuristics_common' if args.common else 'heuristics', args.smoke)
    pipeline_id = 'common_'+args.pipeline if args.common else args.pipeline
    probe_folder = output/'probe'
    probe_id = identity(manifest, pipeline=pipeline_id, stage='probe', method='empirical_displacement',
        dataset='new_probe_episodes', variant='v2', smoke=args.smoke, seeds=[cfg['action_probe_seed']],
        effective={'repetitions': 8}, hashes={'data': None, 'targets': None, 'catalog': None})
    from bayesian_rrtl.comparison import PongEnvironmentFactory
    protocol = pilot_protocol(manifest)
    if args.common:
        from experiments.pong_common_contract import CommonPongConfig, CommonPongFactory
        factory = CommonPongFactory(CommonPongConfig(args.pipeline, cfg['common_protocol']['rom_sha256'],
            cfg['smoke_horizon'] if args.smoke else cfg['common_protocol']['horizon']))
    else:
        factory = PongEnvironmentFactory(AtariConfig(**protocol['environments'][args.pipeline])).with_horizon(
            cfg['smoke_horizon'] if args.smoke else cfg['horizon'])
    with unit_lock(probe_folder):
        unit = WorkUnit(probe_folder, probe_id)
        saved = unit.load()
        if saved is None:
            if args.report_only:
                raise ValueError('No validated action probe; run bounded diagnostics first')
            directions = probe_actions(factory, cfg['action_probe_seed'])
            unit.commit('completed', result=directions)
        else:
            directions = saved['result']
    if not args.report_only:
        for policy in POLICIES:
            for seed in seeds:
                row = play(factory, 'common' if args.common else args.pipeline, policy, seed,
                    cfg['smoke_horizon'] if args.smoke else cfg['horizon'],
                    cfg['max_seconds_per_episode'], directions, folder=output/policy/str(seed),
                    expected=episode_identity(manifest, pipeline_id, policy, seed, args.smoke, directions),
                    extend_seconds=args.extend_seconds, extension_id=args.extension_id)
                print(args.pipeline, policy, seed, row['status'], flush=True)
    result = report(output, pipeline_id, seeds, manifest=manifest, smoke=args.smoke, directions=directions)
    result['directions'] = directions
    result['assisted_gate'] = assisted_gate(result, seeds) if not args.smoke else {'enabled': False, 'reason': 'smoke'}
    atomic_json(output/'report.json', result)
    print(json.dumps({'status': result['status'], 'assisted_gate': result['assisted_gate']}))


if __name__ == '__main__':
    main()
