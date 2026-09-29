"""Frozen independent Pong diagnostics: config, provenance and portable data."""
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np

from bayesian_rrtl.encoding import EncodedBatch, RelationalEncoder
from bayesian_rrtl.replay import Transition

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'pong_bart_analysis_v1'


def source_path(relative):
    """Cluster uses results/; a downloaded read-only local copy uses results_cluster/."""
    path = ROOT / relative
    if path.exists():
        return path
    if relative.startswith('results/'):
        downloaded = ROOT / 'results_cluster' / relative.removeprefix('results/')
        if downloaded.exists():
            return downloaded
    raise FileNotFoundError(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write('\n')


def config(path):
    value = read(path)
    if value.get('schema') != SCHEMA or value.get('pipeline_order') != ['ocatari', 'visual']:
        raise ValueError('Unknown analysis configuration')
    fields = ('development_seeds', 'assisted_collection_seeds', 'fqi_evaluation_seeds')
    seeds = [value['action_probe_seed'], value['smoke_seed'], value['mcmc_seed'], value['bootstrap_seed']]
    for field in fields:
        seeds.extend(value[field])
    if (len(value['development_seeds']) != 20 or len(value['fqi_evaluation_seeds']) != 20
            or len(seeds) != len(set(seeds)) or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds)):
        raise ValueError('Analysis development seeds must be disjoint uint32 values')
    if any(type(value[k]) is not int or value[k] < 1 for k in
           ('horizon', 'smoke_horizon', 'mcmc_burn_in', 'mcmc_draws', 'mcmc_chains',
            'mcmc_seconds_per_job', 'fqi_batch_per_action', 'assisted_max_decisions')):
        raise ValueError('Invalid analysis budget')
    if (value['mcmc_chains'] != 4 or value['mcmc_burn_in'] != 1000 or value['mcmc_draws'] != 2000
            or value['fqi_epochs'] != [1, 5, 10, 20, 40]
            or not 0 < value['fqi_validation_episode_fraction'] < 1
            or not 0 <= value['assisted_random_fraction'] <= 1):
        raise ValueError('Changed diagnostic design requires a new schema/config')
    return value


def verify_source(config_data):
    from experiments.compare_pong import load_frozen, _finals
    campaign = source_path(config_data['source_campaign'])
    manifest = load_frozen(campaign)
    finals = list(_finals(campaign, manifest))
    if len(finals) != 12:
        raise ValueError('The pilot must have twelve complete, validated runs')
    reserved = set(manifest['protocol']['test_seeds'])
    configured = set(config_data['development_seeds'] + config_data['assisted_collection_seeds']
                     + config_data['fqi_evaluation_seeds'] + [config_data[k] for k in
                       ('action_probe_seed', 'smoke_seed', 'mcmc_seed', 'bootstrap_seed')])
    if configured & (reserved | set(manifest['protocol']['training_seeds']) |
                     set(manifest['protocol']['validation_seeds'])):
        raise ValueError('Analysis seeds overlap pilot seeds')
    statuses = []
    for cell, folder, checkpoint, summary in finals:
        rows = read(folder / 'train.json')
        if len(rows) != manifest['protocol']['interactions'] or any(
                row['transition_id'] != i for i, row in enumerate(rows)):
            raise ValueError(f'Incomplete transition log: {cell["id"]}')
        if any(row['episode_seed'] in reserved | configured for row in rows):
            raise ValueError('Analysis/test seed used in pilot training')
        statuses.append({'cell': cell['id'], 'checkpoint_sha256': sha(checkpoint),
                         'interactions': summary['metrics']['interactions'], 'state': 'completed'})
    trace_path = source_path(config_data['source_trace']) / 'report.json'
    trace = read(trace_path)
    target = next(row for row in statuses if row['cell'] == 'visual/bart/73')
    completed_trace = trace.get('traces', {}).get('visual/bart/73', {})
    if (completed_trace.get('status') != 'recorded' or completed_trace.get('checkpoint_sha256') !=
            target['checkpoint_sha256'] or completed_trace.get('steps', 0) < 1):
        raise ValueError('Missing final visual/bart/73 trace')
    return manifest, {'pilot_protocol_sha256': sha(campaign / 'protocol.json'), 'runs': statuses,
        'visual_bart_73_trace_sha256': sha(trace_path), 'visual_bart_73_trace': completed_trace,
        'reserved_test_seeds_untouched': sorted(reserved), 'source_campaign': str(campaign)}


def source_versions():
    from importlib.metadata import PackageNotFoundError, version
    packages = {}
    for name in ('numpy', 'scipy', 'scikit-learn', 'gymnasium', 'ale-py', 'ocatari'):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return {'python': platform.python_version(), 'executable': sys.executable, 'packages': packages}


def source_hashes():
    paths = ('bayesian_rrtl/agent.py', 'bayesian_rrtl/ensemble.py', 'bayesian_rrtl/controls.py',
             'bayesian_rrtl/targets.py', 'bayesian_rrtl/replay.py', 'bayesian_rrtl/diagnostics.py',
             'bayesian_rrtl/games/pong.py', 'experiments/pong_analysis_common.py',
             'experiments/evaluate_pong_heuristics.py', 'experiments/freeze_pong_analysis_data.py',
             'experiments/audit_bart_fixed_data.py', 'experiments/compare_fqi_trees.py', 'experiments/collect_pong_assisted.py',
             'slurm/05_pong_analysis.sbatch', 'configs/pong_bart_analysis_v1.json',
             'requirements-pong-analysis.txt', 'environment-pong-analysis.yml')
    return {p: sha(ROOT / p) for p in paths}


def create_manifest(config_path, output):
    value = config(config_path)
    _, verified = verify_source(value)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    payload = {'schema': SCHEMA, 'config': value, 'config_sha256': sha(config_path),
               'pilot': verified, 'runtime': source_versions(), 'source_hashes': source_hashes()}
    write_new(output / 'manifest.json', payload)
    return payload


def load_manifest(output, *, strict_sources=True):
    value = read(Path(output) / 'manifest.json')
    if value.get('schema') != SCHEMA:
        raise ValueError('Wrong analysis manifest')
    if strict_sources and value['source_hashes'] != source_hashes():
        raise ValueError('Analysis sources changed after freeze')
    return value


def encoded(values, observed, encoder):
    return EncodedBatch(np.asarray(values, dtype=np.int8), np.asarray(observed, dtype=bool),
                        encoder.catalog, encoder.catalog_hash)


def save_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as handle:
        np.savez_compressed(handle, **arrays)
    return sha(path)


def load_npz(path, expected_sha):
    if sha(path) != expected_sha:
        raise ValueError('Portable dataset checksum mismatch')
    return np.load(path, allow_pickle=False)


def episode_split(episode_ids, fraction):
    unique = sorted(set(int(v) for v in episode_ids))
    if len(unique) < 2:
        raise ValueError('Need at least two episodes for holdout')
    n_holdout = max(1, round(len(unique) * fraction))
    holdout = set(unique[-n_holdout:])
    return np.array([int(v) not in holdout for v in episode_ids], dtype=bool), sorted(holdout)


def manifest_encoder(metadata):
    return RelationalEncoder.from_dict(metadata['encoder'])


def transition_from_arrays(data, i, encoder):
    state = encoded(data['state_values'][i:i+1], data['state_observed'][i:i+1], encoder)
    next_state = (encoded(data['next_values'][i:i+1], data['next_observed'][i:i+1], encoder)
                  if data['next_present'][i] else None)
    return Transition(int(data['transition_ids'][i]), state, int(data['actions'][i]),
        float(data['rewards'][i]), float(data['raw_rewards'][i]), next_state,
        bool(data['terminated'][i]), bool(data['truncated'][i]),
        int(data['episode_ids'][i]), int(data['step_ids'][i]))


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/pong_bart_analysis_v1.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    value = config(args.config)
    if args.output.exists():
        raise FileExistsError('New analysis needs a fresh directory')
    manifest = create_manifest(args.config, args.output)
    print(json.dumps({'campaign_id': value['campaign_id'], 'verified_runs': len(manifest['pilot']['runs']),
                      'final_visual_trace': manifest['pilot']['visual_bart_73_trace']['status'],
                      'analysis_output': str(args.output)}, indent=2))


if __name__ == '__main__':
    main()
