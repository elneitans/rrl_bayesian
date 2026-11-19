"""Distinct full-state starts, frozen-policy rollouts and held-out probes.

The new diagnostic evaluation never overwrites official campaign evaluations.
Its horizon describes a continuation from a snapshot, not a complete game.
"""
import copy
import hashlib
import pickle
import time

import numpy as np

from experiments.pong_analysis_runtime import digest
from .baseline_adapter import normalize_facts
from .encoding import RelationalEncoder


def physical_hash(environment):
    """Exclude RNG seeds: distinguish actual Atari RAM plus relational memory."""
    ram = np.asarray(environment.backend.env._env.unwrapped.ale.getRAM(), dtype=np.uint8)
    return digest({'ram': hashlib.sha256(ram.tobytes()).hexdigest(),
                   'relations': environment.relations.snapshot()})


def ram_hash(environment):
    return hashlib.sha256(np.asarray(environment.backend.env._env.unwrapped.ale.getRAM(), dtype=np.uint8).tobytes()).hexdigest()


def encode(agent_or_encoder, state):
    encoder = getattr(agent_or_encoder, 'encoder', agent_or_encoder)
    return encoder.transform([state]).values[0].tolist()


def behavior(state, rng, *, random_fraction=.45):
    """Explicit prefix/probe behavior, independent of the learned policy.

    Actions follow Breakout's registered NOOP/FIRE/RIGHT/LEFT order.
    This is a data collector, not a claim of an optimal heuristic policy.
    """
    if rng.random() < random_fraction:
        return int(rng.integers(4))
    facts = {(f.name, f.obj1, f.obj2): str(f.value) for f in state}
    if facts.get(('present', 'ball_t', None)) != 'True':
        return 1
    position = facts.get(('x', 'player_t', 'ball_t'))
    if position == 'less':
        return 2
    if position == 'more':
        return 3
    return 1  # Explicit launch attempts also when aligned/ball stationary.


def make_start_bank(environment, *, count=6, seed=903711, max_attempts=24, excluded_ram=()):
    """Capture reachable states after logged prefix actions with real diversity."""
    if count < 2:
        raise ValueError('At least two distinct diagnostic starts required')
    encoder = environment.encoder()
    starts, physical, codes, ram_states = [], set(), set(), set()
    forbidden_ram = set(excluded_ram)
    selected_steps = (0, 16, 32, 64, 96, 160, 256, 400)
    for attempt in range(max_attempts):
        episode_seed = seed+attempt
        state = environment.reset(seed=episode_seed)
        rng = np.random.default_rng(np.random.SeedSequence([seed, attempt, 1]))
        actions = []
        for step in range(max(selected_steps)+1):
            if step in selected_steps:
                ph = physical_hash(environment)
                rh = ram_hash(environment)
                code = encode(encoder, state)
                ch = digest(code)
                # First require new relational code; after four, physical novelty suffices.
                if ph not in physical and rh not in ram_states and rh not in forbidden_ram and (ch not in codes or len(codes) >= min(4, count)):
                    starts.append({'id': len(starts), 'reset_seed': episode_seed,
                        'prefix_actions': actions.copy(), 'prefix_decisions': step,
                        'physical_hash': ph, 'ram_sha256': rh, 'encoded_hash': ch, 'encoded_state': code,
                        'state': normalize_facts(state), 'environment': environment.snapshot(),
                        'objects': copy.deepcopy(environment.last_auxiliary)})
                    physical.add(ph); codes.add(ch); ram_states.add(rh)
                    if len(starts) == count:
                        if len(codes) < min(4, count):
                            raise ValueError('Insufficient relational diversity')
                        return {'schema': 'rrtl_distinct_start_bank_v1', 'seed': seed,
                            'starts': starts, 'unique_physical': len(physical),
                            'unique_encoded': len(codes), 'unique_ram': len(ram_states), 'runtime': environment.runtime_manifest()}
            action = 1 if step == 0 else behavior(state, rng)
            state, _, _, terminal, truncated = environment.step(action)
            actions.append(action)
            if terminal or truncated:
                break
    raise ValueError(f'Could not generate {count} diverse real states')


def verify_start(environment, start):
    environment.restore(start['environment'])
    # ALE 0.10.2 restores Stella's full state but not its public getRAM cache;
    # that cache is refreshed only by emulate/processRAM on the next act.
    # Compare the restored system serialization itself, without advancing time.
    # The RAM novelty hashes remain measured live when each start is captured.
    saved = start['environment']
    actual = environment.snapshot()
    actual_system = pickle.dumps(actual['backend']['ale_state'], protocol=4)
    expected_system = pickle.dumps(saved['backend']['ale_state'], protocol=4)
    if (actual_system != expected_system or digest(actual['relations']) != digest(saved['relations'])
            or actual['backend']['objects'] != saved['backend']['objects']
            or actual['steps'] != saved['steps'] or actual['done'] != saved['done']):
        raise ValueError('Snapshot physical/relational continuity mismatch')
    return start['state']


def rollout(environment, agent, start, *, epsilon=0., horizon=3000, seed=904711,
            serve_window=None):
    if not 0 <= epsilon <= 1 or horizon < 1:
        raise ValueError('Invalid diagnostic rollout settings')
    state = verify_start(environment, start)
    encoder = RelationalEncoder(agent.encoder.catalog)  # TimedEncoder must remain immutable too.
    rng = np.random.default_rng(seed)
    counts = dict.fromkeys(agent.actions, 0)
    initial_lives = environment.lives()
    reward_sum = 0.; trace = hashlib.sha256(); seen = set()
    no_ball = life_losses = 0
    serves = []
    if serve_window is not None and start['phase'] in ('initial_serve', 'after_life_loss'):
        serves.append({'phase': start['phase'], 'observed': 0, 'first_fire': None})
    started = time.perf_counter()
    for step in range(horizon):
        q = np.array([root.predict(normalize_facts(state)) for root in agent.root_nodes.values()])
        action = int(rng.integers(len(q))) if rng.random() < epsilon else int(rng.choice(np.flatnonzero(q == q.max())))
        code = encode(encoder, state)
        seen.add(tuple(code)); trace.update(pickle.dumps((code, action), protocol=4))
        counts[agent.actions[action]] += 1
        if serves and serves[-1]['observed'] < serve_window:
            opportunity = serves[-1]
            opportunity['observed'] += 1
            if agent.actions[action] == 'FIRE' and opportunity['first_fire'] is None:
                opportunity['first_fire'] = opportunity['observed']
        no_ball += int(not any(f.name == 'present' and f.obj1 == 'ball_t' and str(f.value) == 'True' for f in state))
        state, _, raw, terminal, truncated = environment.step(action)
        reward_sum += raw
        lost = environment.last_auxiliary['life_lost']
        life_losses += int(lost)
        if terminal or truncated:
            break
        if serve_window is not None and lost and environment.lives() > 0:
            serves.append({'phase': 'after_life_loss', 'observed': 0, 'first_fire': None})
    limited = not terminal and (truncated or step+1 >= horizon)
    return {'raw_return': reward_sum, 'steps': step+1, 'terminated': bool(terminal),
        'truncated': bool(limited), 'return_scope': 'snapshot_continuation',
        'start': start['id'], 'physical_hash': start['physical_hash'], 'epsilon': epsilon,
        'seed': seed, 'horizon': horizon, 'actions': counts, 'life_losses': life_losses,
        'initial_lives': initial_lives, 'final_lives': environment.lives(),
        'no_ball_decisions': no_ball, 'unique_encoded_states': len(seen),
        'trajectory_sha256': trace.hexdigest(), 'seconds': time.perf_counter()-started,
        **({'serve_opportunities': serves} if serve_window is not None else {})}


def validation_probes(environment, bank, *, seed, blocks=8, horizon=128):
    """Fresh behavior-only rollouts, never fed to the training agent.

    Block identity is a short probe, not necessarily a complete game. Random
    streams differ at each maintenance, while starts are shared across variants.
    """
    records = []
    for block in range(blocks):
        start = bank['starts'][block % len(bank['starts'])]
        state = verify_start(environment, start)
        rng = np.random.default_rng(np.random.SeedSequence([seed, block]))
        for step in range(horizon):
            action = behavior(state, rng, random_fraction=.65)
            following, reward, _, terminal, truncated = environment.step(action)
            records.append({'state': normalize_facts(state), 'next_state': normalize_facts(following),
                'action': action, 'reward': reward, 'terminated': bool(terminal),
                'truncated': bool(truncated or (step+1 == horizon and not terminal)), 'block': block})
            state = following
            if terminal or truncated:
                break
    return records
