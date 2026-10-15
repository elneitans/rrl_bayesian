"""Portable representation audit: replay and exact selected transition IDs."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from experiments.pong_analysis_common import load_npz, manifest_encoder, portable_metadata
from experiments.pong_analysis_runtime import array_hash, atomic_json


def counts(values):
    return {str(k): int(v) for k, v in sorted(Counter(values.tolist()).items(), key=lambda kv: str(kv[0]))}


def reward_counts(values):
    return {'positive': int(np.sum(values > 0)), 'negative': int(np.sum(values < 0)),
            'zero': int(np.sum(values == 0))}


def audit_rows(data, encoder, indices, *, convention):
    indices = np.asarray(indices, dtype=int)
    values, observed = data['state_values'][indices], data['state_observed'][indices]
    states = Counter(v.tobytes()+o.tobytes() for v, o in zip(values, observed))
    n = len(indices)
    relations = []
    causes = data['relation_causes'][indices] if 'relation_causes' in data else None
    for j, key in enumerate(encoder.catalog):
        relation = {'key': key.to_dict() if hasattr(key, 'to_dict') else {
            'type': key.type, 'name': key.name, 'obj1': key.obj1, 'obj2': key.obj2},
            'observed_count': int(observed[:, j].sum()),
            'observed_fraction': float(observed[:, j].mean()) if n else None,
            'categories': counts(values[:, j][observed[:, j]]),
            'missing_causes': counts(causes[:, j][~observed[:, j]]) if causes is not None
                              else {'unknown': int((~observed[:, j]).sum())}}
        relations.append(relation)
    above_code, below_code = (2, 0) if convention == 'historical_visual' else (0, 2)
    relevant = {}
    for j, key in enumerate(encoder.catalog):
        if (key.type, key.name, key.obj1, key.obj2) == ('comparative', 'y', 'player_t', 'ball_t'):
            relevant['paddle'] = {'above': int(np.sum(observed[:, j] & (values[:, j] == above_code))),
                'aligned': int(np.sum(observed[:, j] & (values[:, j] == 1))),
                'below': int(np.sum(observed[:, j] & (values[:, j] == below_code))),
                'unavailable': int(np.sum(~observed[:, j]))}
        if key.type == 'comparative' and key.obj1 == 'ball_t' and key.obj2 == 'ball_t-1':
            relevant['ball_motion_'+key.name] = {'categories': counts(values[:, j][observed[:, j]]),
                                                'unavailable': int(np.sum(~observed[:, j]))}
    return {'transitions': n, 'exact_ids': data['transition_ids'][indices].tolist(),
        'ids_sha256': array_hash(data['transition_ids'][indices]),
        'by_action': counts(data['actions'][indices]), 'by_episode': counts(data['episode_ids'][indices]),
        'raw_rewards': reward_counts(data['raw_rewards'][indices]),
        'learning_rewards': reward_counts(data['rewards'][indices]),
        'empty_states': int(np.sum(~observed.any(axis=1))), 'relations': relations,
        'unique_states': len(states), 'largest_state_fraction': max(states.values(), default=0)/n if n else None,
        'top_state_counts': sorted(states.values(), reverse=True)[:10], 'relevant_coverage': relevant,
        'cause_provenance': 'recorded_auxiliary' if causes is not None else 'historical_unknown'}


def collision_examples(data, limit=10):
    """Separate discarded geometry from conditional outcome heterogeneity.

    Only compare evolution at the same encoded state *and action*. Even then,
    stochasticity or unrecorded history can explain it; this is not proof that
    the representation discarded outcome-relevant information.
    """
    if type(limit) is not int or limit < 1:
        raise ValueError('Positive collision example limit required')
    geometry_groups, action_groups = {}, {}
    geometry_examples, evolution_examples = [], []
    has_geometry = 'geometry_json' in data

    def example(old, i, kind):
        same_action = bool(data['actions'][old] == data['actions'][i])
        geometry = [json.loads(data['geometry_json'][j]) for j in (old, i)] if has_geometry else None
        geometry_different = geometry[0] != geometry[1] if has_geometry else None
        evolution_different = (not np.array_equal(data['next_values'][old], data['next_values'][i]) or
                              not np.array_equal(data['next_observed'][old], data['next_observed'][i]) or
                              data['raw_rewards'][old] != data['raw_rewards'][i]) if same_action else None
        value = {'kind': kind, 'ids': [int(data['transition_ids'][j]) for j in (old, i)],
                 'actions': [int(data['actions'][j]) for j in (old, i)],
                 'same_action': same_action, 'geometry_different': geometry_different,
                 'evolution_different': bool(evolution_different) if same_action else None}
        if geometry_different:
            value['geometry'] = geometry
        return value

    for i, (values, observed) in enumerate(zip(data['state_values'], data['state_observed'])):
        key = values.tobytes()+observed.tobytes()
        if has_geometry and len(geometry_examples) < limit:
            old = geometry_groups.setdefault(key, i)
            candidate = example(old, i, 'geometry_collision')
            if candidate['geometry_different']:
                geometry_examples.append(candidate)
        action_key = (key, int(data['actions'][i]))
        old = action_groups.setdefault(action_key, i)
        if len(evolution_examples) < limit:
            candidate = example(old, i, 'same_action_evolution_difference')
            if candidate['evolution_different']:
                evolution_examples.append(candidate)
        if len(geometry_examples) >= limit and len(evolution_examples) >= limit:
            break
    return {'examples': (geometry_examples+evolution_examples)[:limit],
            'geometry_collisions': geometry_examples,
            'same_action_evolution_differences': evolution_examples,
            'example_limit_per_kind': limit,
            'interpretation': 'Geometry collisions show distinct recorded geometry sharing an encoding, not necessarily loss of decision-relevant information. Evolution is compared only for the same action; differences can reflect stochasticity or hidden history and do not by themselves establish aliasing or learning impossibility.'}


def target_stats(targets, lower, upper):
    y = np.asarray(targets, dtype=float)
    return {'samples': len(y), 'sha256': array_hash(y),
        'minimum': float(y.min()) if len(y) else None, 'maximum': float(y.max()) if len(y) else None,
        'median': float(np.median(y)) if len(y) else None,
        'standard_deviation': float(y.std()) if len(y) else None,
        'iqr': float(np.quantile(y, .75)-np.quantile(y, .25)) if len(y) else None,
        'outside_declared_scale_fraction': float(np.mean((y < lower)|(y > upper))) if len(y) else None,
        'declared_scale': [lower, upper]}


def audit(folder, *, convention):
    folder = Path(folder)
    meta = portable_metadata(folder)
    data = load_npz(folder/'fixed_transitions.npz', meta['npz_sha256'])
    encoder = manifest_encoder(meta)
    selected = np.flatnonzero(data['fqi_selected'])
    report = {'schema': 'pong_representation_audit_v2', 'dataset_sha256': meta['npz_sha256'],
        'replay': audit_rows(data, encoder, np.arange(len(data['actions'])), convention=convention),
        'selected': audit_rows(data, encoder, selected, convention=convention),
        'collisions': collision_examples(data)}
    fit_folder = folder.parent/'fixed_fit'
    if fit_folder.exists():
        fit_meta = portable_metadata(fit_folder)
        fit = load_npz(fit_folder/'fixed_fit.npz', fit_meta['npz_sha256'])
        report['fixed_fit_targets'] = {str(i): target_stats(fit[f'a{i}_targets'],
            fit_meta['scale']['lower'], fit_meta['scale']['upper']) for i in range(len(fit_meta['actions']))}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--convention', choices=('historical_visual', 'screen_y_down'), required=True)
    args = parser.parse_args()
    atomic_json(args.output, audit(args.dataset, convention=args.convention))


if __name__ == '__main__':
    main()
