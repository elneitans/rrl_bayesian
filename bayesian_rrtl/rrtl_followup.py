"""Models and a stratified ALE observation bank for the RRTL follow-up."""
import copy
from statistics import mean

import numpy as np

from experiments.pong_analysis_runtime import digest
from .baseline_adapter import normalize_facts
from .revision_diagnostics import behavior, encode, physical_hash, ram_hash
from .stable_revision import StableRevisionAgent
from .targets import bootstrap_mask
from .tree_revision import route, walk

PHASES = ('initial_serve', 'after_life_loss', 'in_play')


class FollowupAgent(StableRevisionAgent):
    @classmethod
    def fork(cls, agent, revision, improvement, *, refit_fraction=1.0):
        if not 0 < refit_fraction <= 1 or (refit_fraction < 1 and improvement.repair != 'refit'):
            raise ValueError('Partial steps require a refit arm and a fraction in (0, 1]')
        result = super().fork(agent, revision, improvement)
        result.refit_fraction = refit_fraction
        return result

    def _refit(self, old, training, targets):
        candidate = super()._refit(old, training, targets)
        if self.refit_fraction < 1:
            for (_, before), (_, after) in zip(walk(old), walk(candidate)):
                after.q_value = before.q_value + self.refit_fraction * (after.q_value - before.q_value)
        return candidate


def make_phase_bank(environment, *, count, seed, max_attempts=48, excluded_ram=()):
    """Capture distinct live states and preserve executable reset/action prefixes."""
    if count < 6 or count % len(PHASES):
        raise ValueError('Use equal coverage of the three Breakout phases')
    quota = count // len(PHASES)
    starts = {phase: [] for phase in PHASES}
    occupied = set()
    excluded = set(excluded_ram)
    encoder = environment.encoder()
    for attempt in range(max_attempts):
        reset_seed = seed + attempt
        rng = np.random.default_rng(np.random.SeedSequence([seed, attempt]))
        state = environment.reset(seed=reset_seed)
        initial_lives = environment.lives()
        prefix = []
        seen_ball = False
        last_loss = None
        wait_before_serve = 14 + attempt % 11
        for step in range(1200):
            facts = {(f.name, f.obj1, f.obj2): str(f.value) for f in state}
            ball = facts.get(('present', 'ball_t', None)) == 'True'
            moving = any(facts.get((axis, 'ball_t', 'ball_t-1')) in ('more', 'less') for axis in ('x', 'y'))
            seen_ball |= ball
            if not ball and not seen_ball and environment.lives() == initial_lives:
                phase = 'initial_serve'
            elif not ball and last_loss is not None:
                phase = 'after_life_loss'
            elif ball and moving:
                phase = 'in_play'
            else:
                phase = None
            # Vary the first prefix between resets and spread in-play starts in time.
            candidate = (phase == 'initial_serve' and step in (0, 3, 7, 11, 13)) or (
                phase == 'after_life_loss' and step - last_loss in (0, 2, 5, 9)) or (
                phase == 'in_play' and step % 13 == 0)
            if candidate and len(starts[phase]) < quota:
                ram = ram_hash(environment)
                if ram not in occupied and ram not in excluded:
                    code = encode(encoder, state)
                    starts[phase].append({
                        'phase': phase, 'reset_seed': reset_seed, 'prefix_actions': prefix.copy(),
                        'prefix_decisions': step, 'physical_hash': physical_hash(environment),
                        'ram_sha256': ram, 'encoded_hash': digest(code), 'encoded_state': code,
                        'state': normalize_facts(state), 'environment': environment.snapshot(),
                        'objects': copy.deepcopy(environment.last_auxiliary),
                    })
                    occupied.add(ram)
                    if len(occupied) == count:
                        bank = [dict(start, id=i) for i, start in enumerate(
                            row for group in starts.values() for row in group)]
                        return {'schema': 'rrtl_phase_bank_v1', 'seed': seed, 'starts': bank,
                                'unique_physical': len({row['physical_hash'] for row in bank}),
                                'unique_encoded': len({row['encoded_hash'] for row in bank}),
                                'unique_ram': len(occupied), 'runtime': environment.runtime_manifest()}
            if phase == 'initial_serve' and step < wait_before_serve:
                action = int(rng.choice((0, 2, 3)))
            elif phase == 'after_life_loss' and step - last_loss < 12:
                action = int(rng.choice((0, 2, 3)))
            else:
                action = behavior(state, rng)
            state, _, _, terminal, truncated = environment.step(action)
            prefix.append(action)
            if terminal or truncated:
                break
            if environment.last_auxiliary['life_lost']:
                last_loss = step + 1
    raise ValueError(f'Incomplete phase coverage: { {phase: len(rows) for phase, rows in starts.items()} }')


def policy_on_start(agent, state):
    values = {action: root.predict(state) for action, root in agent.root_nodes.items()}
    best, second = sorted(values.values(), reverse=True)[:2]
    return {'q': values, 'greedy': [action for action, q in values.items() if q == best],
            'greedy_margin': best - second,
            'routes': {action: route(root, state)[-1][0] for action, root in agent.root_nodes.items()}}


def double_targets(records, online, frozen, *, gamma, bootstrap_on_truncation):
    """Evaluate three backups on exactly the same stored transitions."""
    for row in records:
        mask = bootstrap_mask(row['terminated'], row['truncated'],
                              bootstrap_on_truncation=bootstrap_on_truncation,
                              final_observation_available=row['next_state'] is not None)
        reward = row['reward']
        if not mask:
            yield {'online_max': reward, 'frozen_max': reward, 'double': reward, 'mask': 0}
            continue
        state = row['next_state']
        now = {action: root.predict(state) for action, root in online.items()}
        old = {action: root.predict(state) for action, root in frozen.items()}
        best = max(now.values())
        chosen = [action for action, score in now.items() if score == best]
        yield {'online_max': reward + gamma * best,
               'frozen_max': reward + gamma * max(old.values()),
               'double': reward + gamma * mean(old[action] for action in chosen), 'mask': 1,
               'online_greedy': chosen,
               'frozen_greedy': [action for action, score in old.items() if score == max(old.values())]}
