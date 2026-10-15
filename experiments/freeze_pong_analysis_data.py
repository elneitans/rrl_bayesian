"""Export strict Linux-loaded pilot BART data to immutable portable NPZ/JSON."""
import argparse
from dataclasses import asdict
from pathlib import Path
import json
import io

import numpy as np

from experiments.pong_analysis_common import (ROOT, config, create_manifest, episode_split,
    load_manifest, sha, source_path, source_versions, source_hashes)
from experiments.pong_analysis_runtime import (WorkUnit, atomic_bytes, atomic_json, bytes_hash, unit_lock)


def _transition_arrays(transitions, encoder):
    n, width = len(transitions), len(encoder.catalog)
    values = np.full((n, width), -1, dtype=np.int8)
    observed = np.zeros((n, width), dtype=bool)
    next_values = values.copy()
    next_observed = observed.copy()
    next_present = np.zeros(n, dtype=bool)
    for i, transition in enumerate(transitions):
        values[i] = transition.state.values[0]
        observed[i] = transition.state.observed[0]
        if transition.next_state is not None:
            next_values[i] = transition.next_state.values[0]
            next_observed[i] = transition.next_state.observed[0]
            next_present[i] = True
    return dict(state_values=values, state_observed=observed,
        next_values=next_values, next_observed=next_observed, next_present=next_present,
        transition_ids=np.array([t.transition_id for t in transitions], dtype=np.int64),
        actions=np.array([t.action for t in transitions], dtype=np.int16),
        rewards=np.array([t.reward for t in transitions], dtype=np.float64),
        raw_rewards=np.array([t.raw_reward for t in transitions], dtype=np.float64),
        terminated=np.array([t.terminated for t in transitions], dtype=bool),
        truncated=np.array([t.truncated for t in transitions], dtype=bool),
        episode_ids=np.array([t.episode_id for t in transitions], dtype=np.int64),
        step_ids=np.array([t.step_id for t in transitions], dtype=np.int64))


def fixed_fit(agent, checkpoint, output, *, episode_fraction):
    fit = agent.last_fit
    if fit is None or len(fit.datasets) != len(agent.config.actions):
        raise ValueError('Checkpoint lacks a complete last_fit')
    transitions = {t.transition_id: t for t in agent.replay.transitions}
    arrays = {}
    actions = []
    for action, dataset in enumerate(fit.datasets):
        prefix = f'a{action}_'
        episodes = np.array([transitions[i].episode_id for i in dataset.transition_ids], dtype=np.int64)
        if len(set(episodes)) >= 2:
            train, holdout = episode_split(episodes, episode_fraction)
        else:
            train, holdout = np.ones(len(episodes), dtype=bool), []
        arrays.update({prefix + 'values': dataset.batch.values,
            prefix + 'observed': dataset.batch.observed,
            prefix + 'targets': dataset.targets,
            prefix + 'transition_ids': np.asarray(dataset.transition_ids, dtype=np.int64),
            prefix + 'episode_ids': episodes, prefix + 'regression_train': train})
        calibration = agent.calibrations[action]
        actions.append({'action': agent.config.actions[action], 'index': action,
            'samples': len(dataset.batch), 'data_hash': dataset.data_hash,
            'regression_holdout_episodes': holdout,
            'regression_train_samples': int(train.sum()), 'calibration': asdict(calibration) if calibration else None})
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    npz_hash = bytes_hash(stream.getvalue())
    metadata = {'schema': 'pong_fixed_fit_v1', 'kind': 'fixed_fit',
        'checkpoint_sha256': sha(checkpoint), 'config': agent.config.to_dict(),
        'encoder': agent.encoder.to_dict(), 'fit_id': fit.fit_id,
        'target_model_id': fit.target_model_id, 'interaction': fit.interaction,
        'scale': asdict(agent.config.scale), 'tree_prior': asdict(agent.config.tree_prior),
        'noise_prior_parameters': {'nu': agent.config.nu, 'q': agent.config.noise_quantile,
                                   'variance_floor': agent.config.variance_floor},
        'actions': actions, 'npz_sha256': npz_hash, 'runtime': source_versions(),
        'target_meaning': 'Last pilot fit Bellman targets, never recomputed in this artifact',
        'holdout_limitation': 'Historical target generator may have trained on these episodes; splitting its last batch does not remove generator contamination'}
    return publish_portable(output, metadata, stream.getvalue())


def fixed_transitions(agent, checkpoint, output, *, batch_per_action, fraction, seed,
                      behavior='pilot_replay'):
    transitions = agent.replay.transitions
    if not transitions:
        raise ValueError('Empty replay')
    arrays = _transition_arrays(transitions, agent.encoder)
    from experiments.pong_dataset_selection import partition, select
    train, heldout_mask, partition_meta = partition(arrays, fraction)
    chosen, selection = select(arrays, train, actions=len(agent.config.actions),
                              per_action=batch_per_action, seed=seed)
    holdout = partition_meta['holdout']
    coverage = selection['coverage']
    select_mask = chosen
    arrays['episode_holdout'] = heldout_mask
    arrays['episode_train'] = train
    arrays['fqi_selected'] = select_mask
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    npz_hash = bytes_hash(stream.getvalue())
    metadata = {'schema': 'pong_fixed_transitions_v1', 'kind': 'fixed_transitions',
        'behavior': behavior, 'checkpoint_sha256': sha(checkpoint) if checkpoint else None,
        'encoder': agent.encoder.to_dict(), 'config': agent.config.to_dict(),
        'replay_count': len(transitions), 'fqi_selected_count': int(select_mask.sum()),
        'holdout_episodes': holdout, 'episode_partition': partition_meta, 'selection': selection, 'coverage': coverage,
        'npz_sha256': npz_hash, 'runtime': source_versions(),
        'selection_seed': seed,
        'target_meaning': 'No Bellman labels stored; targets recomputed per learner and epoch'}
    return publish_portable(output, metadata, stream.getvalue())



def publish_portable(output, metadata, raw):
    metadata = {**metadata, 'export_source_hashes': source_hashes()}
    with unit_lock(output):
        unit = WorkUnit(output/'_publication', metadata)
        saved = unit.load()
        if not saved:
            saved = unit.commit('completed', result=metadata, artifacts={'dataset': raw})
        atomic_bytes(output/(metadata['kind']+'.npz'), unit.read_blob(saved['artifacts']['dataset']))
        atomic_json(output/'metadata.json', saved['result'])
        return saved['result']


def export_pilot(specification, output, pipeline):
    from experiments.compare_pong import load_frozen, load_session, _finals
    campaign = source_path(specification['source_campaign'])
    manifest = load_frozen(campaign)
    cell, _, checkpoint, _ = next(item for item in _finals(campaign, manifest)
        if item[0]['id'] == f'{pipeline}/bart/{specification["fixed_fit_seed"]}')
    session = load_session(cell, checkpoint)  # strict runtime/ROM/snapshot check on Linux
    try:
        agent = session.agent
        fit = fixed_fit(agent, checkpoint, output / pipeline / 'fixed_fit',
            episode_fraction=specification['fqi_validation_episode_fraction'])
        transitions = fixed_transitions(agent, checkpoint, output / pipeline / 'fixed_transitions',
            batch_per_action=specification['fqi_batch_per_action'],
            fraction=specification['fqi_validation_episode_fraction'], seed=specification['mcmc_seed'])
        return {'pipeline': pipeline, 'fixed_fit': fit, 'fixed_transitions': transitions,
                'environment': cell['environment'], 'checkpoint_sha256': sha(checkpoint)}
    finally:
        session.environment.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/pong_bart_analysis_v2.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pipeline', choices=('ocatari', 'visual'), required=True)
    args = parser.parse_args()
    specification = config(args.config)
    create_manifest(args.config, args.output)
    load_manifest(args.output, writable=True)
    result = export_pilot(specification, args.output, args.pipeline)
    atomic_json(args.output / args.pipeline / 'freeze_report.json', result)
    print(json.dumps({'pipeline': args.pipeline, 'fixed_fit': result['fixed_fit']['npz_sha256'],
                      'fixed_transitions': result['fixed_transitions']['npz_sha256']}))


if __name__ == '__main__':
    main()
