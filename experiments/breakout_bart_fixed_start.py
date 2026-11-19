"""Fixed-target comparison of cold and carried tree initialization in Breakout."""

from dataclasses import asdict
from io import BytesIO

import numpy as np

from bayesian_rrtl.ensemble import BARTRegressor
from bayesian_rrtl.q_config import BayesianQConfig
from bayesian_rrtl.replay import concatenate_states
from bayesian_rrtl.targets import frozen_q_targets
from experiments.audit_breakout_bart_checkpoints import load_last_fit, reference_panel, summarize
from experiments.pong_analysis_runtime import array_hash, atomic_bytes, checked_write, digest


def frozen_batches(agent):
    batches = [[(ds.batch, ds.targets, {'saved_fit_hash': ds.data_hash, 'ids': ds.transition_ids})
                for ds in agent.last_fit.datasets]]
    rng = np.random.default_rng(np.random.SeedSequence([agent.config.seed, 901]))
    target = agent.target_model
    # Two new draws from retained replay. Keep the original target model fixed
    # across lots and treatments so only the fitted chain's initial state differs.
    for _ in range(2):
        datasets = []
        for action in range(len(agent.config.actions)):
            rows = agent.replay.sample(action, agent.config.batch_size_per_action, rng)
            x = concatenate_states([row.state for row in rows], agent.encoder)
            y = frozen_q_targets(rows, target, agent.encoder, gamma=agent.config.gamma,
                                 bootstrap_on_truncation=agent.config.bootstrap_on_truncation)
            datasets.append((x, y, {'ids': tuple(row.transition_id for row in rows)}))
        batches.append(datasets)
    return batches


def fit_sequence(agent, batches, panel, info, chains, warm_start):
    recipe = BayesianQConfig(**asdict(agent.config), **agent.estimator_config)
    shape = (len(batches), chains, recipe.draws, len(panel), len(recipe.actions))
    q = np.empty(shape)
    traces = {key: np.empty((shape[0], shape[1], shape[2], shape[4]))
              for key in ('sigma2', 'sse', 'leaves', 'max_depth')}
    reuse = np.zeros((shape[0], shape[1], shape[4]), dtype=int)
    seconds = np.zeros_like(reuse, dtype=float)
    for chain in range(chains):
        previous = list(agent.target_model.posteriors)
        for lot, datasets in enumerate(batches):
            for action, (x, y, _) in enumerate(datasets):
                old = agent.model.posteriors[action]
                regressor = BARTRegressor(kernel=recipe.kernel, m=old.m, tree_prior=old.tree_prior,
                                          scale=old.scale, k=old.k, noise_prior=old.noise_prior)
                rng = np.random.default_rng(np.random.SeedSequence([recipe.seed, 177, lot, chain, action]))
                trees = previous[action].draws[-1].trees if warm_start else None
                fitted = regressor.fit(x, y, rng, burn_in=recipe.burn_in, draws=recipe.draws,
                                       initial_trees=trees)
                q[lot, chain, :, :, action] = fitted.predict_latent_draws(panel)
                seconds[lot, chain, action] = fitted.elapsed_seconds
                reuse[lot, chain, action] = fitted.reused_trees or 0
                for name, field in (('sigma2', 'sigma2'), ('sse', 'residual_sum_squares'),
                                    ('leaves', 'total_leaves'), ('max_depth', 'max_depth')):
                    traces[name][lot, chain, :, action] = [getattr(v, field) for v in fitted.trace[recipe.burn_in:]]
                previous[action] = fitted
        print(f'fixed seed={recipe.seed} warm_start={warm_start} chain={chain+1}/{chains}', flush=True)
    published = agent.model.predict(panel)
    old = agent.target_model.predict(panel)
    prior = np.broadcast_to(old, (chains, *old.shape))
    weights = np.array([row['visits'] for row in info])
    records = []
    for lot, values in enumerate(q):
        means = values.mean(axis=1)
        changed = np.any((means == means.max(axis=-1, keepdims=True)) !=
                         (prior == prior.max(axis=-1, keepdims=True)), axis=-1)
        records.append({
            'change_from_previous_weighted_by_panel_visits': np.average(changed, weights=weights, axis=1).tolist(),
            'trees_reused': reuse[lot].tolist(), 'fit_seconds': seconds[lot].tolist(),
            **summarize(values, {name: data[lot] for name, data in traces.items()}, recipe.actions,
                        info, published, old, recipe.scale.lower, recipe.scale.upper),
        })
        prior = means
    return records, {'q': q, 'reused_trees': reuse, 'fit_seconds': seconds, **traces}


def audit(output, source, config):
    for seed in config['fixed_fit_seeds']:
        path = output / 'fixed' / str(seed)
        if (path / 'report.json').exists():
            continue
        agent, ref, _ = load_last_fit(source / 'Breakout' / 'ocatari' / 'bart' / str(seed), seed)
        panel, info = reference_panel(agent)
        batches = frozen_batches(agent)
        meta = [[{**extra, 'x_sha256': array_hash(x.values), 'y_sha256': array_hash(y)}
                 for x, y, extra in lot] for lot in batches]
        reports = {}
        for name in ('control', 'warm_start'):
            reports[name], traces = fit_sequence(agent, batches, panel, info, config['fixed_fit_chains'],
                                                  warm_start=name == 'warm_start')
            stream = BytesIO()
            np.savez_compressed(stream, **traces)
            atomic_bytes(path / (name + '.npz'), stream.getvalue())
        checked_write(path / 'report.json', {'seed': seed, 'source_checkpoint': ref,
                      'target_model_id': agent.target_model.model_id, 'fit_interaction': agent.last_fit.interaction,
                      'first_lot_is_saved_last_fit': True, 'subsequent_lots_are_resampled_from_final_replay': True,
                      'data': meta, 'data_sha256': digest(meta), 'estimator': agent.estimator_config,
                      'reports': reports})
