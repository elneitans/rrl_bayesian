"""Read-only, standard-library audit of all five completed confirmation parents.

Run on the cluster where the immutable checkpoint and evaluation blobs live.
No checkpoint is deserialized and no environment or training is executed.
"""
import argparse
import hashlib
import json
from pathlib import Path

from verify_results import checked, digest


def sha_file(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def artifact(folder, ref, directory):
    path = folder / ref['path']
    assert path.resolve().parent == (folder / directory).resolve(), str(path)
    assert sha_file(path) == ref['sha256'], str(path)
    return path


def verify(parents, config, source_root, output):
    manifest = checked(parents / 'campaign.json')
    assert manifest['schema'] == 'atari_comparison_v3' and not manifest['unresolved']
    raw = dict(manifest['protocol'])
    for key in ('max_decisions', 'milestones', 'evaluation', 'budgets', 'checkpoint_interval'):
        raw.pop(key)
    assert raw == json.loads(config.read_text()), 'Frozen parent protocol differs'
    for name, sha in manifest['source_hashes'].items():
        assert sha_file(source_root / name) == sha, name
    runtime = manifest['runtime']
    assert runtime['platform'] == 'linux' and runtime['python'] == '3.11.7'
    assert all(v == '1' for v in runtime['numeric_threads'].values())
    seeds = list(range(721101, 721106))
    cells = manifest['cells']
    assert len(cells) == 5 and [c['seed'] for c in cells] == seeds
    campaign_sha = digest(manifest)
    results = []
    for cell in cells:
        assert cell['game'] == 'Breakout' and cell['pipeline'] == 'ocatari'
        assert cell['learner'] == 'corrected_common' and cell['estimator']['split_test'] == 'anova'
        folder = parents / cell['id']
        status = checked(folder / 'status.json')
        expected_binding = {'campaign': campaign_sha, 'cell': cell['hash'],
                            'learner': cell['learner'], 'runtime': runtime}
        assert status['binding'] == expected_binding
        assert status['state'] == 'completed' and status['counter'] == 1000000
        assert status['metrics']['interactions'] == 1000000
        ref = status['checkpoint']
        checkpoints = {}
        seen = set()
        while ref:
            assert ref['sha256'] not in seen, 'Checkpoint chain cycle'
            seen.add(ref['sha256'])
            meta = checked(folder / (ref['path'] + '.json'))
            assert meta['checkpoint'] == ref and meta['binding'] == expected_binding
            assert meta['counter'] not in checkpoints, 'Duplicate counter in committed chain'
            checkpoints[meta['counter']] = ref
            if meta['counter'] in (0, 500000, 1000000):
                artifact(folder, ref, 'checkpoints')
            if ref == status['checkpoint']:
                assert meta['counter'] == status['counter'] and meta['log_head'] == status['log_head']
            ref = meta['previous']
        assert set(range(0, 1000001, 25000)) <= checkpoints.keys()
        head = status['log_head']
        expected_end = 1000000
        trace_rows = segments = 0
        while head:
            path = artifact(folder, head, 'logs')
            with path.open() as stream:
                header = json.loads(next(stream))
                assert header['schema'] == 'pong_log_segment_v2'
                assert header['end'] == head['end'] == expected_end
                counter = header['start']
                for line in stream:
                    row = json.loads(line)
                    assert row['interaction'] == counter and row['transition_id'] == counter - 1
                    counter += 1
                    trace_rows += 1
                assert counter == expected_end + 1
            expected_end = header['start'] - 1
            head = header['previous']
            segments += 1
        assert expected_end == 0 and trace_rows == 1000000
        evaluations = []
        schedule = raw['games']['Breakout']['evaluation']
        assert schedule['reserved_test']['checkpoints'] == []
        assert not (folder / 'evaluations' / 'reserved_test').exists()
        for group in ('intermediate', 'final_development'):
            spec = schedule[group]
            for point in spec['checkpoints']:
                for seed in spec['seeds']:
                    unit = folder / 'evaluations' / group / str(point) / str(seed)
                    identity = checked(unit / 'identity.json')
                    expected = {'schema': 'pong_checkpoint_evaluation_v2', 'campaign': campaign_sha,
                                'cell': cell['hash'], 'checkpoint': checkpoints[point], 'group': group,
                                'seeds': spec['seeds'], 'seed': seed, 'horizon': spec['horizon'],
                                'policy': 'greedy_uniform_ties_v1', 'runtime': runtime}
                    assert identity == expected
                    state = checked(unit / 'state.json')
                    assert state['identity_sha256'] == digest(identity) and state['status'] == 'completed'
                    for blob in state['artifacts'].values():
                        assert blob['path'] == blob['sha256'] + '.blob'
                        assert sha_file(unit / 'objects' / blob['path']) == blob['sha256']
                    ep = state['result']
                    assert ep['status'] == 'completed' and ep['seed'] == seed
                    assert 0 < ep['steps'] <= spec['horizon']
                    evaluations.append({'group': group, 'point': point, 'seed': seed,
                                        'raw_return': ep['raw_return'], 'steps': ep['steps']})
        assert len(evaluations) == 11
        results.append({'seed': cell['seed'], 'verified_transitions': trace_rows,
                        'verified_segments': segments, 'evaluations': evaluations,
                        'status_sha256': digest(status)})
        print(json.dumps({'verified_parent': cell['seed'], 'transitions': trace_rows,
                          'evaluations': len(evaluations)}), flush=True)
    result = {'complete': True, 'campaign': campaign_sha, 'parents': results,
              'verified_transitions': sum(r['verified_transitions'] for r in results),
              'development_rollouts': sum(len(r['evaluations']) for r in results),
              'reserved_test_run': False,
              'scope': 'Frozen config/source, committed checkpoint metadata and retained milestone bytes, '
                       'all five million parent log rows, and all55 scheduled development evaluation identities/blobs. '
                       'No model deserialization, emulator replay or claim of algorithmic correctness.'}
    assert result['verified_transitions'] == 5000000 and result['development_rollouts'] == 55
    wrapped = {'schema': 'rrtl_confirmation_parents_verification_v1',
               'payload': result, 'sha256': digest(result)}
    output.write_text(json.dumps(wrapped, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('parents', 'config', 'source-root', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    verify(args.parents, args.config, args.source_root, args.output)
