"""Read-only paired RRTL growth comparison using the existing B/C campaigns."""
import argparse
import copy
from pathlib import Path

from bayesian_rrtl.evaluation import paired_bootstrap
from bayesian_rrtl.online.multigame_protocol import raw_protocol
from bayesian_rrtl.online.protocol import load
from bayesian_rrtl.online.storage import binding, load_checkpoint
from experiments.compare_pong_progressive import consolidate
from experiments.pong_analysis_runtime import atomic_json, digest


def paired_protocols(historical, anova):
    """Reject any scientific/operational difference other than growth and name."""
    stripped = []
    for manifest, mode in ((historical, 'historical_f'), (anova, 'anova')):
        if manifest['schema'] != 'atari_comparison_v3' or manifest['unresolved']:
            raise ValueError('Paired report requires resolved multigame campaigns')
        protocol = copy.deepcopy(raw_protocol(manifest))
        if set(protocol['learners']) != {'corrected_common'}:
            raise ValueError('Pair only the two RRTL growth variants')
        estimator = protocol['learners']['corrected_common']
        if estimator.get('split_test', 'historical_f') != mode:
            raise ValueError('Wrong historical_f/anova campaign order')
        estimator.pop('split_test', None)
        protocol.pop('name')
        stripped.append(protocol)
    if digest(stripped[0]) != digest(stripped[1]):
        raise ValueError('Paired campaigns differ beyond name and split_test')
    if historical['source_hashes'] != anova['source_hashes'] or historical['runtime'] != anova['runtime']:
        raise ValueError('Paired campaigns require identical code and runtime')


def growth(agent):
    actions = {}
    def visit(node):
        yield node
        for child in node.children:
            yield from visit(child)
    for action, root in agent.root_nodes.items():
        nodes = list(visit(root))
        actions[action] = {'leaves': sum(not node.children for node in nodes),
                           'max_depth': max(node.depth for node in nodes)}
    return {'splits': agent.n_splits, 'leaves': sum(v['leaves'] for v in actions.values()),
            'max_depth': max(v['max_depth'] for v in actions.values()), 'by_action': actions}


def report(historical_path, anova_path):
    paths = {'historical_f': Path(historical_path), 'anova': Path(anova_path)}
    manifests = {mode: load(path) for mode, path in paths.items()}
    paired_protocols(manifests['historical_f'], manifests['anova'])
    runs = []
    final_scores = {}
    for mode, output in paths.items():
        manifest = manifests[mode]
        cells = {c['id']: c for c in manifest['cells']}
        consolidated = consolidate(output)
        for row in consolidated['cells']:
            cell = cells[row['cell']]
            final_point = manifest['protocol']['games'][cell['game']]['max_decisions']
            curve, final = [], None
            for evaluation in row.get('evaluations', []):
                # The loader checks checkpoint SHA, contract and committed logs.
                payload, _ = load_checkpoint(output/cell['id'], evaluation['checkpoint'], binding(manifest, cell))
                agent = payload['session']['collector']['agent']
                if agent.config.split_test != mode:
                    raise ValueError('Checkpoint growth criterion differs from paired campaign')
                point = {'decisions': evaluation['counter'], 'group': evaluation['group'],
                         'mean_raw_return': evaluation['report']['mean_raw_return'],
                         'evaluation': evaluation['report'],
                         'training_active_seconds': evaluation['training_active_seconds'],
                         'checkpoint': evaluation['checkpoint'], **growth(agent)}
                if evaluation['group'] == 'intermediate':
                    curve.append(point)
                elif evaluation['group'] == 'final_development' and evaluation['counter'] == final_point:
                    final = point
            record = {'mode': mode, 'game': cell['game'], 'pipeline': cell['pipeline'],
                      'training_seed': cell['seed'], 'cell': cell['id'],
                      'state': row['comparison_state'], 'decisions': row.get('counter', 0),
                      'training_seconds': row.get('training_seconds') if row.get('training_seconds_complete') else None,
                      'training_seconds_recorded_lower_bound': row.get('training_seconds'),
                      'metrics': row.get('metrics'), 'learning_curve': sorted(curve, key=lambda p: p['decisions']),
                      'final_development': final}
            runs.append(record)
            if final and row['training_state'] == 'completed':
                final_scores.setdefault((mode, cell['game'], cell['pipeline']), {})[cell['seed']] = final['mean_raw_return']
    contrasts = []
    for game, pipeline in sorted({(c['game'], c['pipeline']) for c in manifests['anova']['cells']}):
        a = final_scores.get(('anova', game, pipeline), {})
        h = final_scores.get(('historical_f', game, pipeline), {})
        planned = sorted({c['seed'] for c in manifests['anova']['cells'] if (c['game'], c['pipeline']) == (game, pipeline)})
        common = sorted(set(a) & set(h))
        contrasts.append({'game': game, 'pipeline': pipeline, 'difference': 'anova - historical_f',
                          'matched_training_seeds': common,
                          'pending_training_seeds': sorted(set(planned)-set(common)),
                          'per_seed_difference': {s: a[s]-h[s] for s in common},
                          'bootstrap': paired_bootstrap({s: a[s] for s in common}, {s: h[s] for s in common}) if common else None})
    return {'schema': 'rrtl_split_tests_report_v1',
            'campaigns': {mode: {'path': str(paths[mode]), 'hash': digest(m)} for mode, m in manifests.items()},
            'completed': all(r['state'] == 'completed' for r in runs),
            'independent_unit': 'training_seed', 'runs': runs, 'paired_contrasts': contrasts,
            'reserved_test_executed_by_report': False,
            'limitations': ['ANOVA is an experimental growth score on updated Q, not TD targets.',
                            'Temporal dependence, candidate selection and repeated checks invalidate naive false-positive control.',
                            'No raw-score average across games; limited/complete episode scopes are retained.',
                            'Few seeds and degenerate bootstrap intervals do not establish equivalence.',
                            'Unknown attempt costs stay unknown; measured phase time is not parallel calendar time.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--historical', required=True, type=Path)
    parser.add_argument('--anova', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = report(args.historical, args.anova)
    atomic_json(args.output, result)
    print('Paired report:', args.output, 'completed:', result['completed'])
    return 0 if result['completed'] else 3


if __name__ == '__main__':
    raise SystemExit(main())
