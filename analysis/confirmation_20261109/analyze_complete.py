"""Additional descriptive diagnostics after the full confirmation audit passes."""
import argparse
import csv
import json
from pathlib import Path
import statistics as stats

from verify_results import checked, digest

MODES = ['baseline', 'slow_td', 'refit_only']
SEEDS = list(range(721101, 721106))


def save_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(campaign, output):
    manifest = checked(campaign / 'manifest.json')
    report = checked(campaign / 'report.json')
    verified = json.loads((output / 'verified_summary.json').read_text())
    assert verified['complete'] and verified['campaign'] == report['campaign'] == digest(manifest)
    cells = {(c['seed'], c['mode']): c for c in report['continuations']}
    windows = list(csv.DictReader((output / 'training_windows.csv').open()))
    records, training = [], []
    for seed in SEEDS:
        baseline = {e['start']: e['raw_return'] for e in cells[seed, 'baseline']['evaluations']
                    if e['point'] == 500000}
        for mode in MODES:
            cell = cells[seed, mode]
            curve = {point: stats.mean(e['raw_return'] for e in cell['evaluations'] if e['point'] == point)
                     for point in range(0, 500001, 25000)}
            last = {e['start']: e['raw_return'] for e in cell['evaluations'] if e['point'] == 500000}
            initial, final = curve[0], curve[500000]
            best_point = max(curve, key=curve.get)
            peak = max(curve.values())
            running_peak = maximum_drop = 0
            for value in curve.values():
                running_peak = max(running_peak, value)
                maximum_drop = max(maximum_drop, running_peak - value)
            rows = [r for r in windows if int(r['seed']) == seed and r['mode'] == mode]
            assert len(rows) == 100 and sum(int(r['decisions']) for r in rows) == 500000
            n = sum(int(r['complete_new_episodes']) for r in rows)
            weighted = lambda field: sum(float(r[field]) * int(r['decisions']) for r in rows) / 500000
            complete_return_sum = sum(float(r['mean_ended_episode_return']) * int(r['complete_new_episodes'])
                                      for r in rows if int(r['complete_new_episodes']))
            train = {'seed': seed, 'mode': mode, 'complete_new_episodes': n,
                     'mean_complete_episode_return': complete_return_sum / n if n else None,
                     'raw_per_1000_decisions': 1000 * sum(float(r['raw_reward']) for r in rows) / 500000,
                     'mean_abs_td_error': weighted('mean_abs_td_error'),
                     'td_disjoint_argmax_fraction': weighted('disjoint_argmax_before_after_td_fraction'),
                     'fallback_fraction': weighted('fallback_fraction')}
            for start, label in [(0, 'first100k'), (400000, 'last100k')]:
                subset = [r for r in rows if start <= int(r['point_start']) < start + 100000]
                train[label + '_raw_per1000'] = sum(float(r['raw_reward']) for r in subset) / 100
            training.append(train)
            records.append({'seed': seed, 'mode': mode, 'parent_return': initial, 'final_return': final,
                            'change_from_parent': final - initial, 'observed_peak': peak,
                            'peak_at_absolute_counter': 500000 + best_point,
                            'final_drop_from_peak': peak - final, 'maximum_observed_drawdown': maximum_drop,
                            'observed_points_below_parent': sum(v < initial for v in curve.values()),
                            'mean_curve_trapezoidal': sum((curve[p] + curve[p + 25000]) / 2
                                                         for p in range(0, 500000, 25000)) / 20,
                            'starts_better_than_baseline': sum(last[s] > baseline[s] for s in baseline),
                            'starts_equal_to_baseline': sum(last[s] == baseline[s] for s in baseline),
                            'starts_worse_than_baseline': sum(last[s] < baseline[s] for s in baseline),
                            **{k: cell['metrics'][k] for k in ['initial_splits', 'active_splits', 'grows_total',
                                                             'leaves', 'refits_total']}})
    summaries = []
    for mode in MODES:
        chosen = [r for r in records if r['mode'] == mode]
        train = [r for r in training if r['mode'] == mode]
        delta = [r['final_return'] - next(v['final_return'] for v in records
                                        if v['seed'] == r['seed'] and v['mode'] == 'baseline') for r in chosen]
        best_contributor = max(range(5), key=lambda i: delta[i])
        accepted = [e for c in cells.values() if c['mode'] == mode for e in c['events']
                    if e['kind'] == 'refit_review' and e['gate']['accepted']]
        proposals = [e for c in cells.values() if c['mode'] == mode for e in c['events']
                     if e['kind'] == 'refit_review']
        summaries.append({'mode': mode, 'parent_mean': stats.mean(r['parent_return'] for r in chosen),
                          'mean_change_from_parent': stats.mean(r['change_from_parent'] for r in chosen),
                          'seeds_above_parent': sum(r['change_from_parent'] > 0 for r in chosen),
                          'seeds_equal_parent': sum(r['change_from_parent'] == 0 for r in chosen),
                          'seeds_below_parent': sum(r['change_from_parent'] < 0 for r in chosen),
                          'mean_curve_trapezoidal': stats.mean(r['mean_curve_trapezoidal'] for r in chosen),
                          'mean_raw_per1000_decisions': stats.mean(r['raw_per_1000_decisions'] for r in train),
                          'mean_first100k_raw_per1000': stats.mean(r['first100k_raw_per1000'] for r in train),
                          'mean_last100k_raw_per1000': stats.mean(r['last100k_raw_per1000'] for r in train),
                          'mean_td_disjoint_argmax_fraction': stats.mean(r['td_disjoint_argmax_fraction'] for r in train),
                          'largest_net_gain_contributor': SEEDS[best_contributor],
                          'largest_contributor_fraction_of_net_gain': delta[best_contributor] / sum(delta) if sum(delta) else None,
                          'leave_one_seed_out_mean_differences': [(sum(delta) - d) / 4 for d in delta],
                          'grows_total': sum(r['grows_total'] for r in chosen),
                          'refit_proposals': len(proposals), 'refits_accepted': len(accepted),
                          'refits_rejected': len(proposals) - len(accepted),
                          'accepted_sample_mean_mse_increases': sum(e['candidate_mse'] > e['current_mse'] for e in accepted),
                          'accepted_policy_tv_above_unused_limit': sum(e['policy_guard']['mean_tv'] > e['policy_guard']['limit'] for e in accepted),
                          'maximum_accepted_policy_tv': max((e['policy_guard']['mean_tv'] for e in accepted), default=0)})
    save_csv(output / 'per_seed_diagnostics.csv', records)
    save_csv(output / 'training_summary.csv', training)
    result = {'complete': True, 'campaign': digest(manifest), 'summaries': summaries,
              'new_complete_training_episodes': sum(r['complete_new_episodes'] for r in training),
              'limitations': 'Five independent training seeds; starts are related and paired within seed. '
                             'Training-visit diagnostics use different state distributions by method. '
                             'Trapezoidal curve means and leave-one-out contrasts are descriptive supplements, '
                             'not alternate selected endpoints. Refit-only deliberately does not gate on policy TV.'}
    (output / 'extended_summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    analyze(args.campaign, args.output)
