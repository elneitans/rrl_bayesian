"""New common-contract datasets; resumable random or explicitly assisted behavior."""
import argparse
import copy
import io
import json
from pathlib import Path

import numpy as np

from bayesian_rrtl.q_config import BayesianQConfig
from experiments.evaluate_pong_heuristics import ReactivePolicy, assisted_gate, report
from experiments.pong_analysis_common import load_manifest, source_versions
from experiments.pong_analysis_runtime import (WorkUnit, atomic_bytes, atomic_json, bytes_hash,
    digest, episode, extension_arguments, identity, mode_root, unit_lock)
from experiments.pong_common_contract import CommonPongConfig, CommonPongFactory
from experiments.pong_dataset_selection import partition, select


class CollectionController:
    def __init__(self, *, assisted, directions, fraction):
        self.assisted, self.directions, self.fraction = assisted, directions, fraction
        self.reactive = None
        self.records = []
        self.behavior = None
        self.current_aux = None

    def action(self, state, env, rng):
        self.current_aux = copy.deepcopy(env.last_auxiliary)
        if self.reactive is None and self.assisted:
            self.reactive = ReactivePolicy(self.directions['up'], self.directions['down'], env.actions, 'common')
        # Update heuristic memory on EVERY decision, including exploration actions.
        candidate = self.reactive.object_action(env.backend.current_frame) if self.assisted else None
        if not self.assisted or rng.random() < self.fraction:
            self.behavior = 'random'
            return int(rng.integers(len(env.actions)))
        self.behavior = 'object_reactive'
        return candidate

    def transition(self, state, action, next_state, reward, raw, terminal, truncated, env):
        batch, following = env.encoder().transform([state]), env.encoder().transform([next_state])
        self.records.append({'values': batch.values[0].tolist(), 'observed': batch.observed[0].tolist(),
            'next_values': following.values[0].tolist(), 'next_observed': following.observed[0].tolist(),
            'action': action, 'reward': reward, 'raw_reward': raw, 'terminated': terminal,
            'truncated': truncated, 'behavior': self.behavior, 'auxiliary': self.current_aux,
            'next_auxiliary': copy.deepcopy(env.last_auxiliary)})

    def after_step(self, raw):
        if raw and self.reactive:
            self.reactive.reset()


def collected_arrays(rows):
    records, episodes, steps = [], [], []
    for episode_id, row in enumerate(rows):
        for step_id, record in enumerate(row['transitions']):
            records.append(record)
            episodes.append(episode_id)
            steps.append(step_id)
    if not records:
        raise ValueError('No completed collection units')
    arrays = {'state_values': np.array([r['values'] for r in records], dtype=np.int8),
              'state_observed': np.array([r['observed'] for r in records], dtype=bool),
              'next_values': np.array([r['next_values'] for r in records], dtype=np.int8),
              'next_observed': np.array([r['next_observed'] for r in records], dtype=bool),
              'next_present': np.array([not r['terminated'] for r in records], dtype=bool),
              'transition_ids': np.arange(len(records), dtype=np.int64),
              'episode_ids': np.array(episodes, dtype=np.int64), 'step_ids': np.array(steps, dtype=np.int64),
              'behavior_policies': np.array([r['behavior'] for r in records]),
              'relation_causes': np.array([r['auxiliary']['relation_causes'] for r in records]),
              'geometry_json': np.array([json.dumps(r['auxiliary']['objects'], sort_keys=True) for r in records]),
              'history_reset': np.array([r['auxiliary']['history_reset'] for r in records], dtype=bool)}
    for output, field, dtype in [('actions', 'action', np.int64), ('rewards', 'reward', float),
          ('raw_rewards', 'raw_reward', float), ('terminated', 'terminated', bool), ('truncated', 'truncated', bool)]:
        arrays[output] = np.array([r[field] for r in records], dtype=dtype)
    return arrays


def collect(analysis, pipeline, *, assisted=False, smoke=False, extend_seconds=0., extension_id=None):
    manifest = load_manifest(analysis, writable=True)
    cfg = manifest['config']
    contract = cfg['common_protocol']
    factory = CommonPongFactory(CommonPongConfig(pipeline, contract['rom_sha256'], contract['horizon']))
    directions = None
    if assisted:
        # Requires full COMMON-control evidence, never smoke or historical results.
        controls = mode_root(analysis, pipeline, 'heuristics_common', False)
        probe_id = identity(manifest, pipeline='common_'+pipeline, stage='probe', method='empirical_displacement',
            dataset='new_probe_episodes', variant='v2', smoke=False, seeds=[cfg['action_probe_seed']],
            effective={'repetitions': 8}, hashes={'data': None, 'targets': None, 'catalog': None})
        with unit_lock(controls/'probe'):
            saved = WorkUnit(controls/'probe', probe_id).load()
            if not saved or saved['status'] != 'completed':
                raise ValueError('Missing verified common action probe')
            directions = saved['result']
        evidence = report(controls, 'common_'+pipeline, cfg['development_seeds'],
                          manifest=manifest, directions=directions)
        if not assisted_gate(evidence, cfg['development_seeds'])['enabled']:
            raise ValueError('Common assisted collection gate is not satisfied')
    seeds = [cfg['smoke_seed']] if smoke else (cfg['assisted_collection_seeds'] if assisted else cfg['common_collection_seeds'])
    output = mode_root(analysis, pipeline, 'collection_assisted' if assisted else 'collection_common', smoke)
    quota = 16 if smoke else max(1, cfg['assisted_max_decisions']//len(seeds))
    rows = []
    for episode_id, seed in enumerate(seeds):
        expected = identity(manifest, pipeline=pipeline, stage='collection', method='behavior',
            dataset='assisted_common' if assisted else 'new_common', variant='v1', smoke=smoke,
            seeds=[seed, cfg['collection_exploration_seed']+episode_id],
            effective={'contract': contract, 'quota': quota, 'random_fraction': cfg['assisted_random_fraction'] if assisted else 1.},
            hashes={'data': None, 'targets': None, 'catalog': factory.config.manifest()['relations']['catalog_hash']})
        rows.append(episode(factory, CollectionController(assisted=assisted, directions=directions,
            fraction=cfg['assisted_random_fraction']), output/'episodes'/str(seed), expected,
            seed=seed, horizon=quota, seconds=cfg['max_seconds_per_episode'],
            rng_seed=cfg['collection_exploration_seed']+episode_id, administrative_horizon=True,
            extend_seconds=extend_seconds, extension_id=extension_id))
    if any(row['status'] != 'completed' for row in rows):
        atomic_json(output/'report.json', {'status': 'partial', 'episodes': [{k: v for k, v in row.items() if k != 'transitions'} for row in rows]})
        return {'status': 'partial'}
    data = collected_arrays(rows)
    train, holdout, partition_meta = partition(data, cfg['fqi_validation_episode_fraction'])
    selected, selection = select(data, train, actions=6, per_action=cfg['fqi_batch_per_action'], seed=cfg['selection_seed'])
    data.update(episode_train=train, episode_holdout=holdout, fqi_selected=selected)
    stream = io.BytesIO()
    np.savez_compressed(stream, **data)
    learner = BayesianQConfig(actions=factory.config.manifest()['actions'], reward_min=-1., reward_max=1.)
    from bayesian_rrtl.games.pong import PongRelationsV1
    metadata = {'schema': 'pong_fixed_transitions_v2', 'kind': 'fixed_transitions',
        'mode': 'smoke' if smoke else 'full',
        'dataset_kind': 'assisted_common' if assisted else 'new_common', 'common_protocol': contract,
        'encoder': PongRelationsV1().encoder().to_dict(), 'config': learner.to_dict(),
        'npz_sha256': bytes_hash(stream.getvalue()), 'replay_count': len(data['actions']),
        'fqi_selected_count': int(selected.sum()), 'episode_partition': partition_meta,
        'selection': selection, 'selection_seed': cfg['selection_seed'], 'runtime': source_versions(),
        'auxiliary_model_input': False, 'episodes': [{k: v for k, v in row.items() if k not in ('transitions', 'decision_log')} for row in rows]}
    with unit_lock(output/'dataset'):
        unit = WorkUnit(output/'dataset', {'manifest': digest(manifest), 'metadata': metadata | {'runtime': None}})
        saved = unit.load()
        if not saved:
            saved = unit.commit('completed', result=metadata, artifacts={'dataset': stream.getvalue()})
        atomic_bytes(output/'dataset/fixed_transitions.npz', unit.read_blob(saved['artifacts']['dataset']))
        atomic_json(output/'dataset/metadata.json', saved['result'])
    atomic_json(output/'report.json', {'status': 'completed', 'dataset': metadata,
        'limitation': 'Quota/time are administrative; incomplete episodes are excluded from holdout and fitting.'})
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', type=Path, required=True)
    parser.add_argument('--pipeline', choices=('visual', 'ocatari'), required=True)
    parser.add_argument('--assisted', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    extension_arguments(parser)
    args = parser.parse_args()
    print(collect(args.analysis, args.pipeline, assisted=args.assisted, smoke=args.smoke,
                  extend_seconds=args.extend_seconds, extension_id=args.extension_id))


if __name__ == '__main__':
    main()
