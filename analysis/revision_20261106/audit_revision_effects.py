"""Read-only, immediate policy effects of accepted structural revisions.

Compare the saved pre-maintenance roots with the checkpoint immediately after
maintenance, on the exact independent validation probes used by the gate.
No emulator, TD update, refit, new selection, or extra training is performed.
Q predictions are estimates; policy disagreement does not establish true regret.
"""
import argparse
import collections
import gzip
import hashlib
import math
from pathlib import Path
import statistics
import json

from build_report import checked, save_csv


def best_indices(values):
    return {i for i, value in enumerate(values) if value == max(values)}


def compare_predictions(before, after, actions):
    old, new = best_indices(before), best_indices(after)
    # Each learner draws uniformly among exact ties. A distribution change can
    # occur even if the old and new argmax sets still overlap.
    total_variation = .5 * sum(abs((1/len(old) if i in old else 0) -
                                  (1/len(new) if i in new else 0))
                                for i in range(len(actions)))
    return {'disjoint_greedy': int(not old & new),
            'changed_greedy_distribution': int(old != new),
            'greedy_total_variation': total_variation,
            'max_abs_q_change': max(abs(a-b) for a, b in zip(before, after)),
            'old_greedy_margin': sorted(before)[-1]-sorted(before)[-2],
            'old_q_gap_for_new_greedy': max(before)-statistics.mean(before[i] for i in new),
            **{f'{action}_greedy_probability_before': 1/len(old) if i in old else 0
               for i, action in enumerate(actions)},
            **{f'{action}_greedy_probability_after': 1/len(new) if i in new else 0
               for i, action in enumerate(actions)},
            **{f'{action}_greedy_probability_change':
               (1/len(new) if i in new else 0)-(1/len(old) if i in old else 0)
               for i, action in enumerate(actions)}}


def summarize(rows):
    return {'samples': len(rows), **{
        name: statistics.mean(row[name] for row in rows) if rows else None
        for name in (rows[0] if rows else ())}}


def tree_rows(roots):
    from bayesian_rrtl.tree_revision import walk
    return [{'action': action, 'path': path, 'node': node.revision_uid,
             'q': node.q_value, 'literal': None if node.literal is None else str(node.literal)}
            for action, root in roots.items() for path, node in walk(root)]


def compare_prefix(left, right, point, parent_counter):
    """Check whether learning histories were identical before first revision."""
    fields = ('interaction', 'transition_id', 'episode_id', 'step_id', 'episode_seed',
              'epsilon', 'action', 'reward', 'raw_reward', 'terminated', 'truncated',
              'episode_end', 'raw_episode_return', 'td', 'diagnostic')

    def rows(folder):
        status = checked(folder/'status.json')
        expected = parent_counter+1
        for ref in status['traces']:
            if expected > parent_counter+point:
                break
            raw = (folder/ref['path']).read_bytes()
            if hashlib.sha256(raw).hexdigest() != ref['sha256'] or ref['start'] != expected:
                raise ValueError('First-revision prefix trace integrity mismatch')
            for line in gzip.decompress(raw).splitlines():
                row = json.loads(line)
                if row['interaction'] != expected:
                    raise ValueError('First-revision prefix counter mismatch')
                expected += 1
                if row['interaction'] > parent_counter+point:
                    return
                yield row
        if expected <= parent_counter+point:
            raise ValueError('First-revision prefix incomplete')

    a, b = rows(left), rows(right)
    for i in range(point):
        first, second = next(a), next(b)
        for field in fields:
            if first[field] != second[field]:
                return {'exact': False, 'compared_transitions': i,
                        'first_mismatch_counter': first['interaction'], 'first_mismatch_field': field}
    return {'exact': True, 'compared_transitions': point, 'fields': fields}


def audit(campaign, output):
    # These imports require the pinned Linux numerical runtime. Only trusted,
    # hash-verified blobs from our own diagnostic campaign are deserialized.
    from bayesian_rrtl.tree_revision import route
    from bayesian_rrtl.targets import bootstrap_mask
    from experiments.diagnose_rrtl_revision import blob_read

    output.mkdir(parents=True, exist_ok=True)
    report = checked(campaign/'report.json')
    if not report['complete'] or report['missing']:
        raise ValueError('Complete audited campaign required')
    summaries, details = [], []
    cells = {(c['criterion'], c['seed'], c['mode']): c for c in report['continuations']}
    for cell in report['continuations']:
        base = {name: cell[name] for name in ('criterion', 'seed', 'mode')}
        folder = campaign/'continuations'/cell['criterion']/str(cell['seed'])/cell['mode']
        status = checked(folder/'status.json')
        first_accepted = min((event['counter'] for event in cell['events']
                              if event['kind'] in ('prune_review', 'replacement_review') and event['gate']['accepted']),
                             default=None)
        for path in sorted((folder/'reviews').glob('*.json'), key=lambda p: int(p.stem)):
            saved = checked(path)
            if saved['identity'] != status['identity'] or saved['validation_used_for_td']:
                raise ValueError('Validation provenance mismatch')
            accepted = [event for event in saved['reviews'] if event['gate']['accepted']]
            if not accepted:
                continue
            if len(accepted) != 1:
                raise ValueError('Expected at most one installed candidate')
            event = accepted[0]
            reference = blob_read(folder, saved['reference'])
            payload = blob_read(folder, status['checkpoints'][path.stem])
            if reference['counter'] != payload['counter'] or payload['identity'] != status['identity']:
                raise ValueError('Pre/post maintenance frontier mismatch')
            after = payload['session']['collector']['agent']
            before_roots, after_roots = reference['roots'], after.root_nodes
            actions = list(after.actions)
            if list(before_roots) != actions or list(after_roots) != actions:
                raise ValueError('Action order mismatch')
            probes = blob_read(campaign/'holdouts', saved['holdout'])
            if len(probes) != saved['validation_samples']:
                raise ValueError('Validation size mismatch')
            samples, affected, selected, no_ball, with_ball = [], [], [], [], []
            old_losses, new_losses, differences = [], [], collections.defaultdict(list)
            max_other_action_change = 0.
            for row in probes:
                qb = [before_roots[action].predict(row['state']) for action in actions]
                qa = [after_roots[action].predict(row['state']) for action in actions]
                if not all(math.isfinite(x) for x in qb+qa):
                    raise ValueError('Nonfinite Q value')
                sample = compare_predictions(qb, qa, actions)
                samples.append(sample)
                presence = next((fact.value for fact in row['state']
                                 if fact.name == 'present' and fact.obj1 == 'ball_t'), None)
                if presence == 'False':
                    no_ball.append(sample)
                elif presence == 'True':
                    with_ball.append(sample)
                affected_route = any(node.revision_uid == event['node']
                                     for _, node in route(before_roots[event['action']], row['state']))
                if affected_route:
                    affected.append(sample)
                max_other_action_change = max(max_other_action_change,
                    max(abs(qb[i]-qa[i]) for i, action in enumerate(actions) if action != event['action']))
                if not affected_route or actions[row['action']] != event['action']:
                    continue
                selected.append(sample)
                mask = bootstrap_mask(row['terminated'], row['truncated'],
                    bootstrap_on_truncation=after.config.bootstrap_on_truncation,
                    final_observation_available=row['next_state'] is not None)
                target = row['reward'] + after.config.gamma*max(
                    root.predict(row['next_state']) for root in before_roots.values()) if mask else row['reward']
                loss_before = (qb[row['action']]-target)**2
                loss_after = (qa[row['action']]-target)**2
                old_losses.append(loss_before); new_losses.append(loss_after)
                differences[row['block']].append(loss_after-loss_before)
            if max_other_action_change != 0.:
                raise ValueError('Maintenance changed an unrelated action tree')
            if len(selected) != event['gate']['samples']:
                raise ValueError('Gate support not reproduced')
            if not math.isclose(statistics.mean(old_losses), event['current_mse'], rel_tol=1e-10, abs_tol=1e-12):
                raise ValueError('Old predictive loss not reproduced')
            if not math.isclose(statistics.mean(new_losses), event['candidate_mse'], rel_tol=1e-10, abs_tol=1e-12):
                raise ValueError('New predictive loss not reproduced')
            block_difference = statistics.mean(statistics.mean(v) for v in differences.values())
            if not math.isclose(block_difference, event['gate']['mean_difference'], rel_tol=1e-10, abs_tol=1e-12):
                raise ValueError('Paired block loss not reproduced')
            record = {**base, 'point': int(path.stem), 'action': event['action'],
                      'path': event['path'], 'node': event['node'],
                      'old_literal': event['old_literal'], 'new_literal': event['new_literal'],
                      'old_leaves': event['old_leaves'], 'new_leaves': event['new_leaves'],
                      'current_mse': event['current_mse'], 'candidate_mse': event['candidate_mse'],
                      'block_mean_loss_difference': block_difference,
                      'gate_upper_bound': event['gate']['upper_bound'],
                      'gate_threshold': event['gate']['threshold'],
                      'other_action_trees_q_unchanged': True,
                      'reference': saved['reference'], 'checkpoint': status['checkpoints'][path.stem],
                      'all_validation': summarize(samples), 'affected_states': summarize(affected),
                      'gate_action_subset': summarize(selected),
                      'ball_absent': summarize(no_ball), 'ball_present': summarize(with_ball)}
            if event['counter'] == first_accepted:
                control = 'frozen' if cell['mode'] == 'prune' else 'normal'
                prefix = compare_prefix(folder, folder.parent/control, int(path.stem),
                                        status['identity']['parent_counter'])
                control_cell = cells[cell['criterion'], cell['seed'], control]
                control_returns = {row['start']: row['raw_return'] for row in control_cell['evaluations']
                                   if row['point'] == int(path.stem)}
                revised_returns = {row['start']: row['raw_return'] for row in cell['evaluations']
                                   if row['point'] == int(path.stem)}
                if len(control_returns) != report['config']['starts'] or control_returns.keys() != revised_returns.keys():
                    raise ValueError('First-revision evaluation coverage mismatch')
                record['first_intervention'] = {'control_mode': control, 'learning_prefix': prefix,
                    'control_mean_snapshot_return': statistics.mean(control_returns.values()),
                    'revised_mean_snapshot_return': statistics.mean(revised_returns.values()),
                    'paired_start_differences': {start: revised_returns[start]-value
                                                for start, value in control_returns.items()},
                    'scope': 'A single installed revision separates these checkpoints only when the learning prefix matches exactly.'}
            details.append({**record, 'before_trees': tree_rows(before_roots),
                            'after_trees': tree_rows(after_roots)})
            summaries.append({k: v for k, v in record.items() if not isinstance(v, dict)} | {
                scope+'_'+key: value for scope in ('all_validation', 'affected_states', 'gate_action_subset',
                                                  'ball_absent', 'ball_present')
                for key, value in record[scope].items()})
    expected = sum(cell['metrics']['prunes_total']+cell['metrics']['replacements_total']
                   for cell in report['continuations'])
    if len(details) != expected:
        raise ValueError('Not all accepted revisions audited')
    result = {'accepted_revisions': len(details), 'complete': True,
              'scope': 'Immediate pre/post Q and policy effects on gate probes; no learning or emulator advancement',
              'limitations': ['Validation probes are shared with the acceptance gate, not a new independent policy test.',
                  'The Q gap is defined using the old estimated Q, not the true action value or causal regret.',
                  'Only one accepted revision occurs at a maintenance; later TD can change its policy consequences.',
                  'Snapshot returns are evaluated separately; policy disagreement alone establishes no return effect.'],
              'events': details}
    save_csv(output/'revision_policy_effects.csv', summaries)
    (output/'revision_policy_effects.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'accepted_revisions': len(details), 'immediate_loss_reproduced': True, 'complete': True}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    audit(args.campaign, args.output)
