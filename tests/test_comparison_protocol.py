import copy
from dataclasses import asdict
import json
from pathlib import Path

import pytest

from bayesian_rrtl.atari import AtariConfig
from bayesian_rrtl.baseline_adapter import BaselineLearnerConfig
from bayesian_rrtl.comparison_protocol import CONTRASTS, audit_seeds, resolve_protocol
from bayesian_rrtl.q_config import BayesianQConfig
from experiments import compare_pong as runner


def small_protocol():
    baseline = asdict(BaselineLearnerConfig())
    baseline.pop('seed')
    bart = asdict(BayesianQConfig(gamma=.99, epsilon_decay_steps=50000, m=1, burn_in=1, draws=2,
                                 fit_interval=4, warmup_interactions=4, batch_size_per_action=4))
    for field in ('seed', 'actions', 'reward_min', 'reward_max', 'validation_seeds', 'test_seeds'):
        bart.pop(field)
    return {'schema': 'pong_comparison_v1', 'purpose': 'development',
        'environments': {name: AtariConfig(game='Pong', extractor=backend, include_incomplete_states=ram,
                            max_episode_steps=12).resolved_dict()
                         for name, backend, ram in [('visual', 'legacy_visual_pong_v1', False), ('ocatari', 'ocatari_ram_v1', True)]},
        'learners': {'corrected_common': baseline, 'bart': bart}, 'training_seeds': [81],
        'validation_seeds': [330001], 'test_seeds': [430001, 430002], 'interactions': 8,
        'curve_checkpoints': [0, 4, 8], 'evaluation_episodes': 1, 'evaluation_horizon': 6,
        'contrasts': CONTRASTS, 'execution_order_seed': 70, 'primary_checkpoint': 'final',
        'checkpoint_interval': 2, 'budgets': {'training_seconds_per_run': 30, 'evaluation_seconds_total': 120}}


def test_protocol_full_resolution_order_hash_and_no_environment(monkeypatch):
    monkeypatch.setattr('gymnasium.make', lambda *a, **k: pytest.fail('dry-run creates emulator'))
    p = small_protocol()
    p['training_seeds'] = [81, 82, 83]
    resolved = resolve_protocol(p)
    assert len(resolved['cells']) == len(resolved['execution_order']) == 12
    assert resolved == resolve_protocol(p)
    assert [int(key.split('/')[-1]) for key in resolved['execution_order']] == [81]*4 + [82]*4 + [83]*4
    assert len(set(c['hash'] for c in resolved['cells'])) == 12
    for cell in resolved['cells']:
        assert cell['config']['gamma'] == .99
        if cell['learner'] == 'bart':
            assert cell['config']['epsilon_decay_steps'] == 50000
            assert cell['config']['reward_min'] == (-.9 if cell['pipeline'] == 'visual' else -1.)


@pytest.mark.parametrize('field,value', [('schema', 'wrong'), ('training_seeds', [81, 81]),
    ('test_seeds', [330001]), ('curve_checkpoints', [0, 9]), ('evaluation_episodes', 2),
    ('primary_checkpoint', 'best'), ('interactions', 0), ('contrasts', [])])
def test_invalid_protocol_fails_before_runs(tmp_path, field, value):
    p = small_protocol()
    p[field] = value
    with pytest.raises(ValueError):
        runner.freeze(p, tmp_path / 'absent')
    assert not (tmp_path / 'absent').exists()


def test_frozen_protocol_and_source_changed_rejected(tmp_path, monkeypatch):
    output = tmp_path / 'campaign'
    runner.freeze(small_protocol(), output)
    manifest = runner.load_frozen(output)
    monkeypatch.setattr(runner, 'sources', lambda: {'changed': 'hash'})
    with pytest.raises(ValueError, match='source'):
        runner.load_frozen(output)
    assert runner.load_frozen(output, check_sources=False) == manifest
    (output / 'protocol.json').write_text('{}')
    with pytest.raises(ValueError, match='protocol'):
        runner.load_frozen(output, check_sources=False)


def fake_completed_campaign(tmp_path):
    output = tmp_path / 'fake'
    manifest = runner.freeze(small_protocol(), output)
    for cell in manifest['cells']:
        folder = output / cell['id']
        folder.mkdir(parents=True)
        path = folder / 'final_checkpoint.rrtl'
        path.write_bytes(b'synthetic-test-checkpoint')
        runner.write(folder / 'status.json', {'cell_hash': cell['hash'], 'protocol_hash': runner.digest(manifest),
            'state': 'completed', 'interactions': 8, 'checkpoint': path.name, 'checkpoint_sha256': runner.file_hash(path)})
        runner.write(folder / 'summary.json', {'state': 'completed', 'final_checkpoint_sha256': runner.file_hash(path)})
        curve = []
        for point in manifest['protocol']['curve_checkpoints']:
            cp = folder / f'checkpoint_{point}.rrtl'
            cp.write_bytes(path.read_bytes())
            runner.write(folder / f'validation_{point}.json', fake_evaluate(manifest, cell, cp, 'validation'))
            curve.append({'interactions': point, 'mean_return': 1.})
        runner.write(folder / 'summary.json', {'state': 'completed', 'final_checkpoint_sha256': runner.file_hash(path),
                    'curve': curve, 'validation_auc': 1., 'timing': {}, 'metrics': {'complexity': {}}})
    return output, manifest


def fake_evaluate(manifest, cell, checkpoint, split):
    expected = runner.binding(manifest, cell, checkpoint, split, manifest['protocol'][f'{split}_seeds'], 6)
    rows = [{'seed': seed, 'return': 1., 'steps': 6, 'terminated': False, 'truncated': True,
             'return_scope': 'horizon_limited', 'epsilon': 0., 'points_won': 2., 'points_lost': 1.} for seed in expected['seeds']]
    return {'binding': expected, 'report': runner.evaluation_summary(rows, 6), 'seconds': 0.}


def test_partial_test_resume_preserves_complete_and_preflights_all(tmp_path):
    output, manifest = fake_completed_campaign(tmp_path)
    calls = []
    def interrupted(*args):
        calls.append(args[1]['id'])
        if len(calls) == 2:
            raise RuntimeError('interruption')
        return fake_evaluate(*args)
    with pytest.raises(ValueError, match='confirm-test'):
        runner.evaluate_test(output, evaluator=interrupted)
    assert calls == []
    with pytest.raises(RuntimeError, match='interruption'):
        runner.evaluate_test(output, confirmed=True, evaluator=interrupted)
    paths = list(output.glob('*/*/*/test.json'))
    assert len(paths) == 1
    first, before = paths[0], paths[0].read_bytes()
    calls.clear()
    def remaining(*args):
        calls.append(args[1]['id'])
        return fake_evaluate(*args)
    runner.evaluate_test(output, confirmed=True, evaluator=remaining)
    assert len(calls) == 3 and first.read_bytes() == before
    calls.clear()
    runner.evaluate_test(output, confirmed=True, evaluator=remaining)
    assert not calls
    saved = runner.read(first)
    saved['binding']['checkpoint_sha256'] = 'altered'
    runner.write(first, saved)
    last = output / manifest['cells'][-1]['id'] / 'test.json'
    last.unlink()
    with pytest.raises(ValueError, match='binding'):
        runner.evaluate_test(output, confirmed=True, evaluator=remaining)
    assert not calls and not last.exists()


def test_incomplete_run_or_changed_checkpoint_blocks_test_before_seed_use(tmp_path):
    output, manifest = fake_completed_campaign(tmp_path)
    folder = output / manifest['cells'][-1]['id']
    status = runner.read(folder / 'status.json')
    status['state'] = 'partial'
    runner.write(folder / 'status.json', status)
    def forbidden(*args):
        pytest.fail('No reserved seed may be used')
    with pytest.raises(ValueError, match='completed'):
        runner.evaluate_test(output, confirmed=True, evaluator=forbidden)
    status['state'] = 'completed'
    runner.write(folder / 'status.json', status)
    (folder / 'final_checkpoint.rrtl').write_bytes(b'changed')
    with pytest.raises(ValueError, match='checkpoint'):
        runner.evaluate_test(output, confirmed=True, evaluator=forbidden)


def test_atomic_test_write_failure_keeps_previous_results(tmp_path, monkeypatch):
    output, _ = fake_completed_campaign(tmp_path)
    original = runner.write
    def fail(path, data):
        if Path(path).name == 'test.json':
            raise OSError('disk full')
        original(path, data)
    monkeypatch.setattr(runner, 'write', fail)
    with pytest.raises(OSError):
        runner.evaluate_test(output, confirmed=True, evaluator=fake_evaluate)
    assert not list(output.glob('*/*/*/test.json'))


def test_report_test_never_completes_missing_results(tmp_path):
    output, _ = fake_completed_campaign(tmp_path)
    with pytest.raises(FileNotFoundError):
        runner.aggregate(output, 'test')
    assert not list(output.glob('*/*/*/test.json'))


def test_seed_audit_reads_nested_json_and_csv(tmp_path):
    (tmp_path / 'config.json').write_text(json.dumps({'config': {'validation_seeds': [11, 12]}}))
    (tmp_path / 'train.csv').write_text('Episode seed,Reward\n13,14\n')
    result = audit_seeds([tmp_path], [11, 13, 14])
    assert sorted(h['seed'] for h in result['hits']) == [11, 13]
    assert result['files_scanned'] == 2


def test_malformed_evaluation_is_not_accepted(tmp_path):
    output, manifest = fake_completed_campaign(tmp_path)
    cell = manifest['cells'][0]
    saved = fake_evaluate(manifest, cell, output / cell['id'] / 'final_checkpoint.rrtl', 'test')
    for mutate in ('rows', 'nan', 'mean', 'scope'):
        invalid = copy.deepcopy(saved)
        if mutate == 'rows':
            invalid['report']['episodes'].pop()
        elif mutate == 'nan':
            invalid['report']['episodes'][0]['return'] = float('nan')
        elif mutate == 'scope':
            invalid['report']['episodes'][0]['return_scope'] = 'complete_game'
        else:
            invalid['report']['mean_raw_return'] = 100.
        with pytest.raises(ValueError):
            runner.validate_evaluation(invalid, saved['binding'])


def test_operational_budget_extensions_are_explicit_audited_and_do_not_change_protocol(tmp_path):
    output = tmp_path / 'campaign'
    manifest = runner.freeze(small_protocol(), output)
    before = (output / 'protocol.json').read_bytes()
    for kwargs in ({'training_seconds': 30, 'reason': 'same'},
                   {'evaluation_seconds': 119, 'reason': 'lower'},
                   {'training_seconds': float('inf'), 'reason': 'unbounded'},
                   {'training_seconds': float('nan'), 'reason': 'invalid'},
                   {'training_seconds': 40}, {'reason': 'no change'}):
        with pytest.raises(ValueError):
            runner.extend_budgets(output, manifest, **kwargs)
    assert not (output / 'operational_budgets.json').exists()
    runner.extend_budgets(output, manifest, training_seconds=40, reason='Finish checkpointed runs')
    runner.extend_budgets(output, manifest, evaluation_seconds=180, reason='Finish validation')
    limits, revisions = runner.operational_budgets(output, manifest)
    assert limits == {'training_seconds_per_run': 40, 'evaluation_seconds_total': 180}
    assert len(revisions) == 2 and revisions[1]['previous_hash'] == revisions[0]['hash']
    assert revisions[0]['reason'] == 'Finish checkpointed runs'
    assert (output / 'protocol.json').read_bytes() == before
    assert runner.digest(runner.load_frozen(output)) == runner.digest(manifest)
    record = runner.read(output / 'operational_budgets.json')
    record['revisions'][0]['after']['training_seconds_per_run'] = 400
    runner.write(output / 'operational_budgets.json', record)
    with pytest.raises(ValueError, match='history'):
        runner.operational_budgets(output, manifest)


def test_new_evaluation_statistics_and_points_are_validated(tmp_path):
    output, manifest = fake_completed_campaign(tmp_path)
    cell = manifest['cells'][0]
    saved = fake_evaluate(manifest, cell, output / cell['id'] / 'final_checkpoint.rrtl', 'test')
    rows = saved['report']['episodes']
    rows[1].update(points_won=5., points_lost=0., **{'return': 5.})
    saved['report'] = runner.evaluation_summary(rows, 6)
    assert saved['report']['median_raw_return'] == 3.
    assert saved['report']['std_raw_return'] == 2.
    assert saved['report']['iqr_raw_return'] == 2.
    runner.validate_evaluation(saved, saved['binding'])
    for key in ('median_raw_return', 'std_raw_return', 'iqr_raw_return', 'schema'):
        invalid = copy.deepcopy(saved)
        invalid['report'].pop(key)
        with pytest.raises(ValueError):
            runner.validate_evaluation(invalid, saved['binding'])
    for value in (-1, float('nan'), 100):
        invalid = copy.deepcopy(saved)
        invalid['report']['episodes'][0]['points_won'] = value
        with pytest.raises(ValueError, match='point'):
            runner.validate_evaluation(invalid, saved['binding'])


def test_legacy_report_keeps_missing_points_unknown(tmp_path):
    output, manifest = fake_completed_campaign(tmp_path)
    manifest.pop('evaluation_schema')
    runner.write(output / 'protocol.json', manifest)
    (output / 'protocol.sha256').write_text(runner.file_hash(output / 'protocol.json'))
    for cell in manifest['cells']:
        folder = output / cell['id']
        status = runner.read(folder / 'status.json')
        status['protocol_hash'] = runner.digest(manifest)
        runner.write(folder / 'status.json', status)
        for point in manifest['protocol']['curve_checkpoints']:
            saved = fake_evaluate(manifest, cell, folder / f'checkpoint_{point}.rrtl', 'validation')
            for key in ('schema', 'median_raw_return', 'std_raw_return', 'iqr_raw_return'):
                saved['report'].pop(key)
            for row in saved['report']['episodes']:
                row.pop('points_won')
                row.pop('points_lost')
            runner.write(folder / f'validation_{point}.json', saved)
    before = {p: p.read_bytes() for p in output.glob('*/*/*/validation_*.json')}
    report = runner.aggregate(output)
    assert all(r['evaluation_points'] is None and r['median_raw_return'] == 1. for r in report['runs'])
    assert all(p.read_bytes() == data for p, data in before.items())
