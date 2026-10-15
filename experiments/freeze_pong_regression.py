"""Freeze a new common X/y problem with a generator excluding ALL holdout episodes."""
import argparse
from dataclasses import asdict
import io
from pathlib import Path
import pickle

import numpy as np

from bayesian_rrtl.agent import calibration_design
from bayesian_rrtl.priors import calibrate_noise
from bayesian_rrtl.q_config import BayesianQConfig
from bayesian_rrtl.replay import concatenate_states
from bayesian_rrtl.targets import frozen_q_targets
from experiments.compare_fqi_trees import FittedQ, fit_iteration
from experiments.pong_analysis_common import (load_npz, manifest_encoder, transition_from_arrays,
    portable_metadata, source_hashes, source_versions)
from experiments.pong_analysis_runtime import (WorkUnit, array_hash, atomic_bytes, atomic_json,
    bytes_hash, unit_lock)


def freeze(dataset, output, *, generator_epochs=1):
    if type(generator_epochs) is not int or generator_epochs < 1:
        raise ValueError('Positive generator iterations required')
    dataset, output = Path(dataset), Path(output)
    meta = portable_metadata(dataset)
    if 'common_protocol' not in meta:
        raise ValueError('New independent regression requires common-protocol data')
    data = load_npz(dataset/'fixed_transitions.npz', meta['npz_sha256'])
    encoder = manifest_encoder(meta)
    train_indices = np.flatnonzero(data['fqi_selected'] & data['episode_train'])
    if not len(train_indices) or not data['episode_holdout'].any():
        raise ValueError('Independent regression requires complete training and holdout episodes')
    train = tuple(transition_from_arrays(data, int(i), encoder) for i in train_indices)
    config = BayesianQConfig(**meta['config'])
    provenance = {'code': source_hashes(), 'runtime': source_versions()}
    previous = FittedQ(config.actions, (None,)*len(config.actions), encoder.catalog_hash)
    previous_hash = bytes_hash(pickle.dumps(previous))
    for epoch in range(1, generator_epochs+1):
        expected = {'dataset': meta['npz_sha256'], 'catalog': encoder.catalog_hash,
            'provenance': provenance,
            'method': 'cart', 'epoch': epoch, 'config': meta['config'],
            'hashes': {'previous_model': previous_hash},
            'train_ids': array_hash(data['transition_ids'][train_indices]),
            'targets': array_hash(frozen_q_targets(train, previous, encoder, gamma=config.gamma,
                                                  bootstrap_on_truncation=config.bootstrap_on_truncation))}
        previous, record = fit_iteration(output/'generator'/str(epoch), expected, 'cart', train,
                                     encoder, meta['config'], previous, epoch)
        previous_hash = record['model_sha256']
    included = np.flatnonzero(data['fqi_selected'] | data['episode_holdout'])
    transitions = tuple(transition_from_arrays(data, int(i), encoder) for i in included)
    targets = frozen_q_targets(transitions, previous, encoder, gamma=config.gamma,
                               bootstrap_on_truncation=config.bootstrap_on_truncation)
    arrays, actions = {}, []
    for action in range(len(config.actions)):
        rows = np.array([i for i, t in enumerate(transitions) if t.action == action], dtype=int)
        if not len(rows):
            raise ValueError('No selected/holdout data for an action')
        batch = concatenate_states([transitions[i].state for i in rows], encoder)
        is_train = data['episode_train'][included[rows]]
        if not is_train.any():
            raise ValueError('Action missing training data; collect more before freezing')
        calibration = calibrate_noise(config.scale.normalize(targets[rows][is_train]),
            calibration_design(batch.take(np.flatnonzero(is_train))), nu=config.nu,
            quantile=config.noise_quantile, variance_floor=config.variance_floor)
        prefix = f'a{action}_'
        arrays.update({prefix+'values': batch.values, prefix+'observed': batch.observed,
            prefix+'targets': targets[rows], prefix+'transition_ids': data['transition_ids'][included[rows]],
            prefix+'episode_ids': data['episode_ids'][included[rows]], prefix+'regression_train': is_train})
        actions.append({'action': config.actions[action], 'data_hash': array_hash(batch.values),
                        'calibration': asdict(calibration)})
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    metadata = {'schema': 'pong_fixed_fit_v2', 'kind': 'fixed_fit', 'mode': meta.get('mode', 'full'),
        'dataset_kind': meta.get('dataset_kind', 'new_common')+'_fixed_regression',
        'source_dataset': meta['npz_sha256'], 'common_protocol': meta['common_protocol'],
        'generator_method': 'cart', 'generator_epochs': generator_epochs,
        'generator_train_ids': data['transition_ids'][train_indices].tolist(),
        'config': meta['config'], 'encoder': meta['encoder'], 'actions': actions,
        'scale': asdict(config.scale), 'tree_prior': asdict(config.tree_prior),
        'noise_prior_parameters': {'nu': config.nu, 'q': config.noise_quantile, 'variance_floor': config.variance_floor},
        'npz_sha256': bytes_hash(stream.getvalue()),
        'holdout_limitation': 'Whole holdout episodes excluded from ALL generator fits and calibration; fixed y shared by estimators.'}
    with unit_lock(output/'problem'):
        unit = WorkUnit(output/'problem', {'dataset': meta['npz_sha256'], 'generator_epochs': generator_epochs,
                                          'config': meta['config'], 'catalog': encoder.catalog_hash,
                                          'provenance': provenance})
        saved = unit.load()
        if not saved:
            saved = unit.commit('completed', result=metadata, artifacts={'data': stream.getvalue()})
        atomic_bytes(output/'fixed_fit.npz', unit.read_blob(saved['artifacts']['data']))
        atomic_json(output/'metadata.json', saved['result'])
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--generator-epochs', type=int, default=1)
    args = parser.parse_args()
    print(freeze(args.dataset, args.output, generator_epochs=args.generator_epochs)['npz_sha256'])


if __name__ == '__main__':
    main()
