"""Read-only partial campaign review; never labels missing runs complete."""
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path
import statistics as stats


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def checked(path):
    raw = json.loads(path.read_text())
    if raw['schema'] != 'pong_analysis_unit_v2' or digest(raw['payload']) != raw['sha256']:
        raise ValueError(f'Invalid checksum/schema: {path}')
    return raw['payload']


def save_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader()
        writer.writerows(rows)


def review(root, output, trace_root=None):
    output.mkdir(parents=True, exist_ok=True)
    manifest = checked(root/'manifest.json')
    campaign = digest(manifest)
    config = manifest['config']
    starts = {s['id']: s for s in checked(root/'starts.json')}
    assert len(starts) == config['starts'] == 12
    banks = [list(starts.values()), checked(root/'validation_starts.json')]
    banks.extend(checked(p) for p in sorted((root/'return_starts').glob('*.json')))
    ram = set()
    for bank in banks:
        new = {s['ram_sha256'] for s in bank}
        assert len(new) == len(bank) and not ram & new
        ram.update(new)
    units, evaluations, curves, reviews, events, windows = [], [], [], [], [], []
    available = {}
    for group in manifest['groups']:
        parent = next(p for p in manifest['parents'] if (p['criterion'], p['seed']) == (group['criterion'], group['seed']))
        for mode in config['modes']:
            base = {**group, 'mode': mode}
            folder = root/'continuations'/group['criterion']/str(group['seed'])/mode
            identity = {'campaign': campaign, 'parent': parent['checkpoint'], 'criterion': group['criterion'],
                        'seed': group['seed'], 'mode': mode, 'parent_counter': parent['counter']}
            if not (folder/'status.json').exists():
                units.append({**base, 'state': 'not_started', 'new_decisions': 0, 'evaluation_rollouts': 0})
                continue
            status = checked(folder/'status.json')
            assert status['identity'] == identity and status['counter'] == config['parent_counter'] + status['continuation_decisions']
            frontier = config['parent_counter']
            for ref in status['traces']:
                assert ref['start'] == frontier+1
                frontier = ref['end']
            assert frontier == status['counter']
            es = []
            for path in sorted((folder/'evaluations').glob('*/*/state.json')):
                point, start_id = map(int, path.parts[-3:-1])
                assert str(point) in status['checkpoints'] and start_id in starts
                expected = {**identity, 'checkpoint': status['checkpoints'][str(point)], 'point': point,
                            'start_bank': manifest['start_bank'], 'start': start_id, 'epsilon': 0.,
                            'horizon': config['evaluation_horizon'], 'evaluation_seed': config['evaluation_seed']+start_id}
                ei, saved = checked(path.with_name('identity.json')), checked(path)
                assert ei == expected and saved['identity_sha256'] == digest(ei)
                if saved['status'] != 'completed':
                    continue
                result = saved['result']
                assert result['start'] == start_id and result['seed'] == expected['evaluation_seed']
                assert result['epsilon'] == 0 and result['horizon'] == config['evaluation_horizon']
                assert result['physical_hash'] == starts[start_id]['physical_hash']
                assert 0 < result['steps'] <= config['evaluation_horizon']
                assert sum(result['actions'].values()) == result['steps']
                es.append({'point': point, **result})
                evaluations.append({**base, 'point': point, **result})
            for point in sorted({v['point'] for v in es}):
                values = [v for v in es if v['point'] == point]
                if len(values) != len(starts):
                    continue
                assert len({v['start'] for v in values}) == len(starts)
                returns = [v['raw_return'] for v in values]
                steps = sum(v['steps'] for v in values)
                curves.append({**base, 'point': point, 'mean_return': stats.mean(returns),
                    'median_return': stats.median(returns), 'minimum_return': min(returns), 'maximum_return': max(returns),
                    'starts': len(values), 'truncated': sum(v['truncated'] for v in values),
                    'fire_fraction': sum(v['actions'].get('FIRE', 0) for v in values)/steps,
                    'no_ball_fraction': sum(v['no_ball_decisions'] for v in values)/steps})
            unit_events = []
            for path in sorted((folder/'audits').glob('*.json'), key=lambda p: int(p.stem)):
                if path.stem not in status['checkpoints']:
                    continue
                audit = checked(path)
                assert audit['identity'] == identity and audit['counter'] == config['parent_counter']+int(path.stem)
                unit_events.extend(audit['events'])
            for kind, key in [('grow', 'grows_total'), ('target_sync', 'target_syncs'), ('prune_review', 'prunes_total'),
                              ('replacement_review', 'replacements_total'), ('refit_review', 'refits_total')]:
                n = sum(e['kind'] == kind and (kind in ('grow', 'target_sync') or e['gate']['accepted']) for e in unit_events)
                assert n == status['metrics'][key], (base, kind, n, status['metrics'][key])
            events.extend({**base, **e} for e in unit_events)
            for path in sorted((folder/'reviews').glob('*.json'), key=lambda p: int(p.stem)):
                if path.stem not in status['checkpoints']:
                    continue
                saved = checked(path)
                assert saved['identity'] == identity and not saved['validation_used_for_td'] and not saved['evaluation_bank_used_for_gate']
                for e in saved['reviews']:
                    if e['gate']['accepted'] and mode in ('guarded', 'stable_guarded'):
                        assert all(e[k]['accepted'] for k in ('loss_gate', 'policy_guard', 'return_guard'))
                        assert e['policy_guard']['mean_tv'] <= config['variants'][mode]['max_policy_tv']
                    if 'return_guard' in e:
                        gate = e['return_guard']
                        assert not gate['training_updated'] and not gate['evaluation_bank_used']
                        before, after = gate['baseline'], gate['candidate']
                        assert len(before) == len(after) == config['return_starts']
                        diff = [b['raw_return']-a['raw_return'] for a,b in zip(before, after)]
                        assert diff == gate['differences'] and abs(stats.mean(diff)-gate['mean_difference']) < 1e-10
                    reviews.append({**base, 'point': int(path.stem), **e})
            units.append({**base, 'state': status['state'], 'new_decisions': status['continuation_decisions'],
                          'evaluation_rollouts': len(es), **status['metrics']})
            available[group['criterion'], group['seed'], mode] = es
            if trace_root is not None:
                from experiments.improvement_trace_summary import trace_summary
                trace_folder = trace_root/folder.relative_to(root)
                windows.extend({**base, **w} for w in trace_summary(trace_folder, status, config['parent_counter']))
    final = [r for r in curves if r['point'] == config['continuation_decisions']]
    matched = []
    for row in final:
        key = row['criterion'], row['seed']
        new = {v['start']: v['raw_return'] for v in available[key+(row['mode'],)] if v['point'] == config['continuation_decisions']}
        old = {v['start']: v['raw_return'] for v in available[key+('baseline',)] if v['point'] == config['continuation_decisions']}
        assert set(new) == set(old) == set(starts)
        diffs = [new[i]-old[i] for i in sorted(starts)]
        matched.append({**{k: row[k] for k in ('criterion', 'seed', 'role', 'mode')},
                        'mean_difference': stats.mean(diffs), 'median_difference': stats.median(diffs),
                        'wins': sum(d>0 for d in diffs), 'ties': sum(d==0 for d in diffs), 'losses': sum(d<0 for d in diffs)})
    pairwise = []
    for criterion in ('historical_f', 'anova'):
        for mode in config['modes']:
            ms = [r for r in matched if r['criterion']==criterion and r['mode']==mode]
            seeds = {v['seed'] for v in ms}
            if not seeds:
                continue
            candidate = [r['mean_return'] for r in final if r['criterion']==criterion and r['mode']==mode and r['seed'] in seeds]
            baseline = [r['mean_return'] for r in final if r['criterion']==criterion and r['mode']=='baseline' and r['seed'] in seeds]
            pairwise.append({'criterion':criterion, 'mode':mode, 'seeds':len(seeds),
                'candidate_mean':stats.mean(candidate), 'matched_baseline_mean':stats.mean(baseline),
                'mean_paired_difference':stats.mean(v['mean_difference'] for v in ms),
                'seed_wins':sum(v['mean_difference']>0 for v in ms),
                'seed_ties':sum(v['mean_difference']==0 for v in ms),
                'seed_losses':sum(v['mean_difference']<0 for v in ms)})
    cohorts, summaries = {}, []
    for criterion in ('historical_f', 'anova'):
        seeds = [s for s in config['seeds'] if all(any(r['criterion']==criterion and r['seed']==s and r['mode']==mode for r in final) for mode in config['modes'])]
        cohorts[criterion] = seeds
        for mode in config['modes']:
            rs = [r for r in final if r['criterion']==criterion and r['seed'] in seeds and r['mode']==mode]
            ds = [r for r in matched if r['criterion']==criterion and r['seed'] in seeds and r['mode']==mode]
            if not rs:
                continue
            summaries.append({'criterion': criterion, 'mode': mode, 'seeds': len(seeds),
                'mean_seed_return': stats.mean(r['mean_return'] for r in rs),
                'median_seed_return': stats.median(r['mean_return'] for r in rs),
                'mean_paired_difference': stats.mean(r['mean_difference'] for r in ds),
                'seed_wins': sum(r['mean_difference']>0 for r in ds),
                'seed_ties': sum(r['mean_difference']==0 for r in ds),
                'seed_losses': sum(r['mean_difference']<0 for r in ds)})
    reasons = collections.Counter()
    for e in reviews:
        reason = 'accepted' if e['gate']['accepted'] else e['gate'].get('reason', 'loss_rejected')
        reasons[e['mode'], reason] += 1
    summary = {'campaign': campaign, 'campaign_complete': False, 'planned_continuations': 70,
        'states': dict(collections.Counter(u['state'] for u in units)),
        'recorded_new_decisions': sum(u['new_decisions'] for u in units),
        'checked_evaluation_rollouts': len(evaluations), 'complete_final_panels': len(final),
        'common_complete_cohorts': cohorts, 'method_summary': summaries, 'pairwise_available_summary':pairwise,
        'operations': {k: sum(u.get(k,0) for u in units) for k in ['grows_total','prunes_total','replacements_total','refits_total','target_syncs']},
        'review_outcomes': [{'mode':m,'reason':k,'count':v} for (m,k),v in sorted(reasons.items())],
        'validation_scope': 'Checksummed manifest/status/evaluation identities, disjoint RAM banks, event counters and recorded gates. No emulator/checkpoint deserialization; no independent reconstruction of candidate losses/targets here.',
        'trace_rows_verified': sum(w['decisions'] for w in windows),
        'limitations': 'Partial prospective campaign; dependent early starts, development plus limited confirmation seeds. No reserved test, no winner selection or general superiority claim.'}
    for name, rows in [('units',units),('evaluations',evaluations),('curves',curves),('final',final),('matched_vs_baseline',matched),('common_cohort_summary',summaries),('pairwise_available_summary',pairwise),('training_windows',windows)]:
        save_csv(output/(name+'.csv'), rows)
    (output/'reviews.json').write_text(json.dumps(reviews,indent=2)+'\n')
    (output/'events.json').write_text(json.dumps(events,indent=2)+'\n')
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--trace-root',type=Path)
    args=parser.parse_args()
    review(args.campaign,args.output,args.trace_root)
