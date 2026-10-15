"""Deterministic, episode-isolated selections; no artificial reward replication."""
import argparse
from pathlib import Path

import numpy as np

from experiments.pong_analysis_common import load_npz, portable_metadata
from experiments.pong_analysis_runtime import array_hash


def complete_episodes(data):
    result = []
    for episode in sorted(set(data['episode_ids'].tolist())):
        indices = np.flatnonzero(data['episode_ids'] == episode)
        steps = data['step_ids'][indices]
        if (len(steps) and steps[0] == 0 and np.array_equal(steps, np.arange(len(steps)))
                and bool(data['terminated'][indices[-1]] or data['truncated'][indices[-1]])):
            result.append(int(episode))
    return result


def partition(data, fraction=.2):
    episodes = complete_episodes(data)
    if not 0 < fraction < 1:
        raise ValueError('Invalid holdout fraction')
    if len(episodes) < 2:
        # No independent episode holdout is possible; report, do not invent one.
        return np.isin(data['episode_ids'], episodes), np.zeros(len(data['actions']), dtype=bool), {
            'status': 'insufficient_complete_episodes', 'complete': episodes, 'holdout': [],
            'excluded_incomplete': sorted(set(data['episode_ids'].tolist())-set(episodes))}
    count = max(1, min(len(episodes)-1, round(len(episodes)*fraction)))
    heldout = episodes[-count:]
    train = np.isin(data['episode_ids'], episodes[:-count])
    holdout = np.isin(data['episode_ids'], heldout)
    return train, holdout, {'status': 'frozen', 'complete': episodes, 'holdout': heldout,
        'excluded_incomplete': sorted(set(data['episode_ids'].tolist())-set(episodes))}


def select(data, train, *, actions, per_action, seed, rule='uniform_action'):
    if type(per_action) is not int or per_action < 1 or rule not in ('uniform_action', 'stratified_reward'):
        raise ValueError('Selection must explicitly declare 128/512 and its rule')
    rng = np.random.default_rng(seed)
    selected = np.zeros(len(train), dtype=bool)
    coverage = []
    for action in range(actions):
        candidates = np.flatnonzero(train & (data['actions'] == action))
        n = min(per_action, len(candidates))
        if rule == 'uniform_action':
            chosen = rng.choice(candidates, n, replace=False)
        else:
            # Round-robin among reward strata, redistribution when a stratum is depleted.
            pools = [list(rng.permutation(candidates[np.sign(data['raw_rewards'][candidates]) == sign]))
                     for sign in (-1, 0, 1)]
            chosen = []
            while len(chosen) < n:
                for pool in pools:
                    if pool and len(chosen) < n:
                        chosen.append(pool.pop())
            chosen = np.asarray(chosen, dtype=int)
        selected[chosen] = True
        coverage.append({'action': action, 'available': len(candidates), 'selected': n,
                         'insufficient_samples': len(candidates) < per_action,
                         'positive_count': int(np.sum(data['raw_rewards'][chosen] > 0)),
                         'no_positive_signal': not bool(np.any(data['raw_rewards'][chosen] > 0))})
    return selected, {'rule': rule, 'per_action': per_action, 'selection_seed': seed,
        'selected_ids_sha256': array_hash(data['transition_ids'][selected]), 'coverage': coverage,
        'distribution_changed': rule == 'stratified_reward', 'duplicates': False}


def freeze_selection(dataset, output, *, per_action, seed, rule):
    dataset, output = Path(dataset), Path(output)
    meta = portable_metadata(dataset)
    arrays = load_npz(dataset/'fixed_transitions.npz', meta['npz_sha256'])
    data = {name: arrays[name] for name in arrays.files}
    # Never repartition a previously frozen dataset in a selection sensitivity.
    if 'episode_holdout' not in data:
        raise ValueError('Dataset needs a frozen complete-episode partition first')
    selected, details = select(data, data['episode_train'], actions=len(meta['config']['actions']),
                              per_action=per_action, seed=seed, rule=rule)
    data['fqi_selected'] = selected
    expected = {'source': meta['npz_sha256'], 'selection': details}
    from experiments.freeze_pong_analysis_data import publish_portable
    import io
    from experiments.pong_analysis_runtime import bytes_hash
    stream = io.BytesIO()
    np.savez_compressed(stream, **data)
    result = {**meta, 'npz_sha256': bytes_hash(stream.getvalue()), 'selection_identity': expected,
              'selection': details, 'fqi_selected_count': int(selected.sum())}
    return publish_portable(output, result, stream.getvalue())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--per-action', choices=(128, 512), type=int, default=128)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--rule', choices=('uniform_action', 'stratified_reward'), default='uniform_action')
    args = parser.parse_args()
    print(freeze_selection(args.dataset, args.output, per_action=args.per_action, seed=args.seed, rule=args.rule)['selection'])


if __name__ == '__main__':
    main()
