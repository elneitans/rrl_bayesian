"""Discardable B12 process/signal recovery and B11 reference-cache benchmark.

Default environment is simulated. --atari uses the common RGB/RAM contract and
requires the original verified Linux runtime. Neither path can execute reserved test.
"""
import argparse
from dataclasses import asdict
import json
import multiprocessing as mp
import os
from pathlib import Path
import signal
import sys
import time

import numpy as np

from bayesian_rrtl.encoding import RelationKey, RelationalEncoder
from bayesian_rrtl.online.learners import LEARNERS, PredictionCache
from bayesian_rrtl.online.protocol import freeze, load, resolve, ROOT
from bayesian_rrtl.online.storage import binding, load_checkpoint, stream_rows
from experiments.compare_pong_progressive import (StopRequest, consolidate, default_factory,
                                                 evaluate_cell, train_cell)
from experiments.pong_analysis_runtime import atomic_json, checked_read, unit_lock
from preprocessing import Fact


class SimulatedEnvironment:
    actions = ('NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE')

    def __init__(self, horizon=128):
        self.horizon = horizon
        self.steps = 0
        self.seed = None

    def encoder(self):
        return RelationalEncoder([RelationKey('logical', 'present', 'ball')])

    def state(self):
        return [Fact('logical', str(bool(self.steps%2)), 'present', 'ball')]

    def reset(self, *, seed):
        self.seed, self.steps = seed, 0
        return self.state()

    def step(self, action):
        self.steps += 1
        raw = (1. if action == 2 else -1.) if self.steps%4 == 0 else 0.
        return self.state(), raw, raw, self.steps == 7, False

    def snapshot(self):
        return {'steps': self.steps, 'seed': self.seed}

    def restore(self, state):
        self.steps, self.seed = state['steps'], state['seed']

    def runtime_manifest(self):
        return {'kind': 'discardable_simulated_pong_api_v1', 'horizon': self.horizon}

    def close(self):
        pass


class SimulatedFactory:
    def __init__(self, horizon=128):
        self.horizon = horizon

    def __call__(self):
        return SimulatedEnvironment(self.horizon)

    def with_horizon(self, horizon):
        return SimulatedFactory(horizon)


def _worker(output, selection, atari, request_after=None, notifications=None, continue_event=None):
    cell = load(output)['cells'][int(selection)]
    factory = default_factory(cell) if atari else SimulatedFactory(cell['environment']['horizon'])
    with StopRequest() as stop:
        def after_publish(counter):
            if request_after is not None and counter >= request_after:
                notifications.put({'pid': os.getpid(), 'checkpoint': counter})
                # Parent sends a real USR1 while the process is between committed updates.
                if not continue_event.wait(30):
                    raise TimeoutError('Recovery coordinator did not release worker')
        train_cell(output, selection, factory=factory, stop=stop,
                   after_publish=after_publish if request_after is not None else None)


def artifacts(output, cell):
    m = load(output)
    folder = Path(output)/cell['id']
    status = checked_read(folder/'status.json')
    data, _ = load_checkpoint(folder, status['checkpoint'], binding(m, cell))
    rows = list(stream_rows(folder, status['log_head']))
    return data, rows, status


def model_signature(agent, learner):
    """Exact learned structure/parameters; exclude clocks and query instrumentation."""
    if not LEARNERS[learner].batch:
        def node(n):
            return {'q': n.q_value, 'literal': repr(n.literal), 'children': [node(c) for c in n.children]}
        return {name: node(n) for name, n in agent.root_nodes.items()}
    result = []
    for model in agent.model.posteriors:
        if model is None:
            result.append(None)
        elif learner == 'cart':
            result.append(asdict(model.tree))
        elif learner == 'bart':
            result.append({'draws': [asdict(d) for d in model.draws],
                           'scale': asdict(model.scale), 'data_hash': model.data_hash})
        else:
            result.append([{'children_left': t.tree_.children_left.tolist(),
                            'children_right': t.tree_.children_right.tolist(),
                            'feature': t.tree_.feature.tolist(), 'threshold': t.tree_.threshold.tolist(),
                            'value': t.tree_.value.tolist()} for t in model.estimator.estimators_])
    return result


def compare_exact(left, right, cell):
    a, x, _ = artifacts(left, cell)
    b, y, _ = artifacts(right, cell)
    if len(x) != len(y) or [{k:v for k,v in r.items() if k != 'fit'} for r in x] != [
            {k:v for k,v in r.items() if k != 'fit'} for r in y]:
        raise AssertionError('Continuous/resumed decisions or episodes differ')
    ac, bc = a['session']['collector'], b['session']['collector']
    assert ac['exploration_rng'].bit_generator.state == bc['exploration_rng'].bit_generator.state
    assert ac['environment_rng'].bit_generator.state == bc['environment_rng'].bit_generator.state
    assert model_signature(ac['agent'], cell['learner']) == model_signature(bc['agent'], cell['learner'])
    if LEARNERS[cell['learner']].batch:
        assert ac['agent'].replay_rng.bit_generator.state == bc['agent'].replay_rng.bit_generator.state
        for one, two in zip(ac['agent'].mcmc_rngs, bc['agent'].mcmc_rngs):
            assert one.bit_generator.state == two.bit_generator.state
        for one, two in zip(ac['agent'].replay.transitions, bc['agent'].replay.transitions):
            assert one.transition_id == two.transition_id
            np.testing.assert_array_equal(one.state.values, two.state.values)
            np.testing.assert_array_equal(one.state.observed, two.state.observed)
    return {'decisions': len(x), 'ended_episodes': sum(r['episode_end'] for r in x),
            'exact_decisions_models_rng': True, 'no_duplicate_transitions': len({r['transition_id'] for r in y}) == len(y)}


def cache_benchmark(agent, repetitions=200):
    batch = agent.encoder.transform([[], []])
    expected = agent.model.predict(batch)
    measurements = []
    for limit in (0, 1048576):
        cache = PredictionCache(limit)
        start = time.perf_counter()
        for _ in range(repetitions):
            np.testing.assert_array_equal(cache.predict(agent.model, batch), expected)
        measurements.append({'cache_limit_bytes': limit, 'seconds': time.perf_counter()-start,
                             'accounted_bytes': cache.bytes, 'hits': cache.hits, 'misses': cache.misses})
    return {'repetitions': repetitions, 'rows_per_prediction': len(batch), 'exact_prediction': True,
            'same_fixed_model_and_batch': True, 'measurements': measurements,
            'limitation': 'Microbenchmark includes exact-array assertions; repeated states do not represent Atari hit rate.'}


def validate(output, *, atari=False, selection=None):
    if atari and sys.platform != 'linux':
        raise ValueError('Atari recovery requires the original Linux environment')
    output = Path(output)
    if output.exists():
        raise FileExistsError('Use a new discardable recovery folder')
    raw = json.loads((ROOT/'configs/pong_online_smoke.json').read_text())
    cells = resolve(raw)['cells']
    indices = [int(selection)] if selection is not None else list(range(len(cells)))
    if any(i < 0 or i >= len(cells) for i in indices):
        raise ValueError('Invalid recovery cell')
    freeze(raw, output/'continuous')
    freeze(raw, output/'resumed')
    ctx = mp.get_context('spawn')
    report = {'environment': 'Linux Atari common contract' if atari else 'simulated', 'cells': [],
              'reserved_test_executed': False}
    for index in indices:
        cell = cells[index]
        factory = default_factory(cell) if atari else SimulatedFactory(cell['environment']['horizon'])
        notifications, proceed = ctx.Queue(), ctx.Event()
        child = ctx.Process(target=_worker, args=(output/'resumed', index, atari, 16, notifications, proceed))
        child.start()
        notice = notifications.get(timeout=60)
        os.kill(notice['pid'], signal.SIGUSR1)
        # Let the Python signal handler run before releasing the next safe boundary.
        time.sleep(.05)
        proceed.set()
        child.join(60)
        if child.is_alive():
            child.terminate()
            child.join(10)
            raise TimeoutError('Interrupted worker did not stop')
        if child.exitcode != 0:
            raise RuntimeError('Interrupted worker failed')
        _, _, partial = artifacts(output/'resumed', cell)
        assert partial['state'] == 'partial' and partial['reason'] == 'interrupted_signal'
        continued = ctx.Process(target=_worker, args=(output/'resumed', index, atari))
        continued.start()
        continued.join(120)
        if continued.is_alive():
            continued.terminate()
            continued.join(10)
            raise TimeoutError('Resume exceeded bounded smoke')
        if continued.exitcode != 0:
            raise RuntimeError('Resumed worker failed')
        train_cell(output/'continuous', index, factory=factory)
        exact = compare_exact(output/'continuous', output/'resumed', cell)
        folder = output/'resumed'/cell['id']
        with unit_lock(folder):
            try:
                train_cell(output/'resumed', index, factory=factory)
            except RuntimeError as exc:
                assert 'Concurrent' in str(exc)
            else:
                raise AssertionError('Second writer was accepted')
        evaluate_cell(output/'resumed', index, factory=factory)
        views = {p: p.read_bytes() for p in (folder/'evaluations').rglob('state.json')}
        evaluate_cell(output/'resumed', index, factory=factory)
        assert all(p.read_bytes() == before for p, before in views.items())
        saved, _, _ = artifacts(output/'resumed', cell)
        agent = saved['session']['collector']['agent']
        entry = {'cell': cell['id'], **exact, 'signal_recovery': True, 'second_writer_rejected': True,
                 'completed_evaluation_reused': True}
        if LEARNERS[cell['learner']].batch:
            entry['cache_benchmark'] = cache_benchmark(agent)
        report['cells'].append(entry)
    report['reconstructed'] = consolidate(output/'resumed')
    report['continuous_state'] = consolidate(output/'continuous')
    report['note'] = 'Selected-cell Linux run leaves other cells pending; use all cells to validate full smoke.'
    atomic_json(output/'recovery_report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--atari', action='store_true')
    parser.add_argument('--cell')
    args = parser.parse_args()
    result = validate(args.output, atari=args.atari, selection=args.cell)
    print(json.dumps({'report': str(args.output/'recovery_report.json'),
                      'environment': result['environment'], 'verified_cells': len(result['cells'])}, indent=2))


if __name__ == '__main__':
    main()
