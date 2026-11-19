"""Compare BART sampling budgets on exact last-fit data from trusted Breakout checkpoints."""

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
from importlib.metadata import version
from pathlib import Path
import pickle
import platform

import numpy as np

from bayesian_rrtl.diagnostics import chain_diagnostics
from bayesian_rrtl.encoding import EncodedBatch
from bayesian_rrtl.ensemble import BARTRegressor
from bayesian_rrtl.online.storage import checkpoint_metadata, verified_bytes
from bayesian_rrtl.q_config import BayesianQConfig
from experiments.pong_analysis_runtime import atomic_json, checked_read


def load_last_fit(folder, seed):
    status = checked_read(folder / 'status.json')
    ref = status['checkpoint']
    meta = checkpoint_metadata(folder, ref)
    payload = pickle.loads(verified_bytes(folder, ref))  # trusted campaign only
    for key in ('binding', 'counter', 'log_head'):
        if payload[key] != meta[key] or payload[key] != status[key]:
            raise ValueError(f'Inconsistent checkpoint {key}')
    agent = payload['session']['collector']['agent']
    fit = agent.last_fit
    if (payload['counter'] != 1_000_000 or agent.interactions != payload['counter']
            or agent.learner_id != 'bart' or agent.config.seed != seed
            or fit is None or fit.fit_id != agent.model.model_id
            or fit.target_model_id != agent.target_model.model_id):
        raise ValueError('Expected a matching 1M Breakout BART checkpoint and last fit')
    for action, dataset in enumerate(fit.datasets):
        x = dataset.batch
        ids = np.asarray(dataset.transition_ids, dtype='<i8')
        y = np.asarray(dataset.targets, dtype='<f8')
        digest = hashlib.sha256(x.catalog_hash.encode() + ids.tobytes() + x.values.tobytes()
                                + x.observed.tobytes() + y.tobytes()).hexdigest()
        if (dataset.action != action or len(x) == 0 or len(x) != len(ids) or len(x) != len(y)
                or not np.isfinite(y).all() or digest != dataset.data_hash):
            raise ValueError(f'Invalid saved fit dataset for action {action}')
    if len(fit.datasets) != len(agent.config.actions):
        raise ValueError('Missing saved action datasets')
    return agent, ref, payload['binding']['runtime']


def reference_panel(agent):
    rows = Counter(tuple(t.state.values[0]) for t in agent.replay.transitions)
    resets = Counter(tuple(t.state.values[0]) for t in agent.replay.transitions if t.step_id == 0)
    if not resets:
        raise ValueError('No episode reset states in saved replay')
    others = sorted(rows.keys() - resets.keys(), key=lambda row: (-rows[row], row))[:16]
    keys = sorted(resets) + others
    values = np.array(keys, dtype=np.int8)
    batch = EncodedBatch(values, values != -1, agent.encoder.catalog, agent.encoder.catalog_hash)
    info = [{'row': list(map(int, key)), 'visits': rows[key], 'reset_visits': resets[key]} for key in keys]
    return batch, info


def sampling(agent, panel, chains, end, seed):
    recipe = BayesianQConfig(**asdict(agent.config), **agent.estimator_config)
    trace_names = ('sigma2', 'sse', 'leaves', 'max_depth')
    traces = {name: np.empty((chains, end, len(recipe.actions))) for name in trace_names}
    q = np.empty((chains, end, len(panel), len(recipe.actions)))
    timings = np.empty((chains, len(recipe.actions)))
    for chain in range(chains):
        for action, dataset in enumerate(agent.last_fit.datasets):
            old = agent.model.posteriors[action]
            regressor = BARTRegressor(kernel=recipe.kernel, m=old.m, tree_prior=old.tree_prior,
                                     scale=old.scale, k=old.k, noise_prior=old.noise_prior)
            rng = np.random.default_rng(np.random.SeedSequence([seed, chain, action]))
            posterior = regressor.fit(dataset.batch, dataset.targets, rng, burn_in=0, draws=end)
            q[chain, :, :, action] = posterior.predict_latent_draws(panel)
            timings[chain, action] = posterior.elapsed_seconds
            for name, field in zip(trace_names, ('sigma2', 'residual_sum_squares', 'total_leaves', 'max_depth')):
                traces[name][chain, :, action] = [getattr(sweep, field) for sweep in posterior.trace]
        print(f'seed={seed} chain={chain + 1}/{chains} fit_seconds={timings[chain].sum():.1f}', flush=True)
    return q, traces, timings


def diagnostic(values):
    result = chain_diagnostics(values)
    result['mcse_bulk_proxy'] = (float(values.std() / np.sqrt(result['ess_bulk']))
                                 if result['ess_bulk'] else None)
    return result


def summarize(q, traces, actions, panel_info, published, previous, lower, upper):
    chain_means = q.mean(axis=1)
    pooled = chain_means.mean(axis=0)
    fire = actions.index('FIRE')
    margin = q[..., fire] - np.max(np.delete(q, fire, axis=-1), axis=-1)

    def greedy(values):
        return [actions[i] for i in np.flatnonzero(values == values.max())]

    result = []
    for i, info in enumerate(panel_info):
        competitors = [j for j in range(len(actions)) if j != fire]
        competitor = max(competitors, key=lambda j: pooled[i, j])
        gap = q[:, :, i, fire] - q[:, :, i, competitor]
        result.append({**info, 'published_q': published[i].tolist(), 'previous_q': previous[i].tolist(),
                       'published_greedy': greedy(published[i]), 'previous_greedy': greedy(previous[i]),
                       'pooled_q': pooled[i].tolist(), 'pooled_greedy': greedy(pooled[i]),
                       'chain_q': chain_means[:, i].tolist(),
                       'chain_greedy': [greedy(row) for row in chain_means[:, i]],
                       'fire_draw_win_fraction_by_chain': np.mean(margin[:, :, i] > 0, axis=1).tolist(),
                       'fire_top_competitor': actions[competitor],
                       'fire_gap_per_chain_mean': gap.mean(axis=1).tolist(),
                       'fire_gap_quantiles': np.quantile(gap, [.05, .5, .95]).tolist(),
                       'fire_gap_diagnostics': diagnostic(gap),
                       'q_diagnostics': {action: diagnostic(q[:, :, i, j])
                                         for j, action in enumerate(actions)}})
    return {'panel': result, 'draw_q_outside_scale_fraction': float(np.mean((q < lower) | (q > upper))),
            'action_diagnostics': {action: {name: diagnostic(values[:, :, j])
                                            for name, values in traces.items()}
                                   for j, action in enumerate(actions)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='Trusted root containing Breakout/ocatari/bart')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, choices=(710602, 710608), required=True)
    parser.add_argument('--chains', type=int, default=4)
    parser.add_argument('--burn-in', type=int, default=200)
    parser.add_argument('--draws', type=int, default=300)
    args = parser.parse_args()
    if args.chains < 2 or args.draws < 4 or args.burn_in < 0:
        parser.error('Require >=2 chains, >=4 draws, and nonnegative burn-in')
    folder = args.source / 'Breakout' / 'ocatari' / 'bart' / str(args.seed)
    agent, ref, historical_runtime = load_last_fit(folder, args.seed)
    panel, info = reference_panel(agent)
    actions = tuple(agent.config.actions)
    lower, upper = agent.config.scale.lower, agent.config.scale.upper
    published, previous = agent.model.predict(panel), agent.target_model.predict(panel)
    short_burn, short_draws = (agent.estimator_config[key] for key in ('burn_in', 'draws'))
    end = args.burn_in + args.draws
    if end < short_burn + short_draws or short_draws < 4:
        parser.error('The extended run must include the production sampling window')
    output = args.output / str(args.seed)
    output.mkdir(parents=True, exist_ok=False)
    q, traces, timings = sampling(agent, panel, args.chains, end, args.seed)
    np.savez_compressed(output / 'traces.npz', q=q, fit_seconds=timings, **traces)
    windows = {'production': slice(short_burn, short_burn + short_draws),
               'extended': slice(args.burn_in, end)}
    report = {name: {'burn_in': window.start, 'draws': window.stop - window.start,
                     **summarize(q[:, window], {k: v[:, window] for k, v in traces.items()},
                                 actions, info, published, previous, lower, upper)}
              for name, window in windows.items()}
    fit = agent.last_fit
    atomic_json(output / 'report.json', {'seed': args.seed, 'chains': args.chains,
        'actions': actions, 'checkpoint': ref, 'fit_id': fit.fit_id, 'target_model_id': fit.target_model_id,
        'fit_interaction': fit.interaction, 'estimator': agent.estimator_config, 'bounds': [lower, upper],
        'historical_runtime': historical_runtime,
        'audit_runtime': {'python': platform.python_version(), 'platform': platform.system(),
                          'machine': platform.machine(), 'numpy': np.__version__, 'scipy': version('scipy')},
        'fit_seconds': timings.tolist(), 'catalog': [asdict(key) for key in panel.catalog],
        'datasets': {action: {'rows': len(ds.batch), 'hash': ds.data_hash,
                              'target_range': [float(ds.targets.min()), float(ds.targets.max())],
                              'outside_scale': int(np.count_nonzero((ds.targets < lower) | (ds.targets > upper)))}
                     for action, ds in zip(actions, fit.datasets)}, 'windows': report})


if __name__ == '__main__':
    main()
