"""Three-seed paired Breakout pilot, isolated cost measurements and fixed-data diagnostics."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def run(config_path, output):
    config_path, output = Path(config_path).resolve(), Path(output).resolve()
    raw = json.loads(config_path.read_text())
    seeds = raw['training_seeds']
    if len(seeds) < 3 or len(set(seeds)) != len(seeds):
        raise ValueError('Pilot requires at least three distinct training seeds')
    output.mkdir(parents=True, exist_ok=False)
    (output / 'config.json').write_text(json.dumps(raw, indent=2))
    runs = []
    for seed in seeds:
        pair = {'seed': seed}
        for variant in ('bayesian', 'corrected'):
            destination = output / f'seed_{seed}' / variant
            print(json.dumps({'starting_seed': seed, 'variant': variant}), flush=True)
            subprocess.run([sys.executable, str(ROOT / 'train_atari.py'), '--config', str(config_path),
                            '--variant', variant, '--seed', str(seed), '--output', str(destination)], check=True)
            summary = json.loads((destination / 'summary.json').read_text())
            if summary['interactions'] != raw['interactions'] or not summary['finite_data']:
                raise ValueError('Failed count/finite-data integrity check')
            if (destination / 'test.json').exists():
                raise ValueError('Pilot must not evaluate reserved test seeds')
            pair[variant] = summary
        a, b = pair['bayesian'], pair['corrected']
        common = min(len(a['training_seeds']), len(b['training_seeds']))
        if a['training_seeds'][:common] != b['training_seeds'][:common]:
            raise ValueError('Episode-seed streams are not paired')
        if [r['seed'] for r in a['evaluation']] != [r['seed'] for r in b['evaluation']]:
            raise ValueError('Validation seeds are not paired')
        pair['validation_mean_difference'] = float(np.mean([r['return'] for r in a['evaluation']]) -
                                                   np.mean([r['return'] for r in b['evaluation']]))
        runs.append(pair)
        (output / 'runs.json').write_text(json.dumps(runs, indent=2, allow_nan=False))
    # Predeclared first training seed, not whichever run achieves the best return.
    diagnostics = raw['diagnostics']
    checkpoint = output / f'seed_{seeds[0]}/bayesian/final_checkpoint.rrtl'
    subprocess.run([sys.executable, '-m', 'experiments.diagnose_atari', '--checkpoint', str(checkpoint),
                    '--output', str(output / 'diagnostics'), '--chains', str(diagnostics['chains']),
                    '--burn-in', str(diagnostics['burn_in']), '--draws', str(diagnostics['draws']),
                    '--seed', str(diagnostics['seed'])], cwd=ROOT, check=True)
    diagnostic_report = json.loads((output / 'diagnostics/report.json').read_text())
    report = {'milestone': 'H6', 'environment': raw['environment'], 'config': raw, 'runs': runs,
              'diagnostics': diagnostic_report,
              'median_training_seconds': {v: float(np.median([r[v]['training_seconds'] for r in runs]))
                                          for v in ('bayesian', 'corrected')},
              'mean_paired_validation_difference': float(np.mean([r['validation_mean_difference'] for r in runs])),
              'extrapolation_hours_2m_interactions': {
                  v: float(np.median([r[v]['training_seconds'] / raw['interactions'] * 2_000_000 / 3600 for r in runs]))
                  for v in ('bayesian', 'corrected')},
              'limitations': [
                  'OCAtari is not installed or implemented: this pilot uses the historical visual extractor.',
                  'Three short seeds establish pipeline operation/cost, not comparative effectiveness.',
                  'Episode limits and batch/MCMC budgets are reduced; inference is approximate unless diagnostics pass.',
                  'Extrapolation includes profiling overhead and assumes fixed cost; replay coverage and tree complexity change.',
                  'Baseline shares extraction but retains historical tree representation and incremental learning.',
                  'No test-seed evaluation or hyperparameter selection from test returns.']}
    (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({'report': str(output / 'report.json'), 'seeds': seeds,
                      'median_training_seconds': report['median_training_seconds']}), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/bayesian_breakout_pilot.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.config, args.output)
