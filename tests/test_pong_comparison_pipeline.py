import copy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from bayesian_rrtl.comparison_protocol import ROOT
from experiments import compare_pong as runner
from test_comparison_protocol import small_protocol


@pytest.mark.atari
@pytest.mark.ocatari
def test_four_cells_cli_and_completed_resume_is_nonmutating(tmp_path):
    protocol = tmp_path / 'protocol.json'
    protocol.write_text(json.dumps(small_protocol()))
    output = tmp_path / 'campaign'
    def command(*args):
        result = subprocess.run([sys.executable, '-m', 'experiments.compare_pong', *map(str, args)],
                                cwd=ROOT, capture_output=True, text=True, timeout=90)
        assert result.returncode == 0, result.stdout + result.stderr
    command('--protocol', protocol, '--dry-run')
    assert not output.exists()
    command('--protocol', protocol, '--output', output, '--stage', 'train')
    status = runner.read(output / 'campaign_status.json')
    assert set(status['states'].values()) == {'completed'}
    manifest = runner.load_frozen(output)
    before = {str(path): path.read_bytes() for cell in manifest['cells']
              for path in (output / cell['id']).iterdir() if path.is_file()}
    command('--output', output, '--stage', 'resume')
    for path, data in before.items():
        assert Path(path).read_bytes() == data
    assert not list(output.glob('*/*/*/test.json'))
    report = runner.read(output / 'report_validation.json')
    assert set(report['contrasts']) == {'visual', 'ocatari'}
    assert len(report['runs']) == 4


@pytest.mark.atari
@pytest.mark.parametrize('learner', ['bart', 'corrected_common'])
@pytest.mark.parametrize('pipeline', ['visual', pytest.param('ocatari', marks=pytest.mark.ocatari)])
def test_interrupted_validation_resume_only_missing_and_exact_continuation(tmp_path, monkeypatch, learner, pipeline):
    output = tmp_path / 'campaign'
    manifest = runner.freeze(small_protocol(), output)
    cell = next(c for c in manifest['cells'] if c['pipeline'] == pipeline and c['learner'] == learner)
    folder = output / cell['id']
    original = runner.evaluate_saved
    calls = []
    def fail_at_four(manifest, cell, path, split, **kwargs):
        calls.append(path.name)
        if path.name == 'checkpoint_4.rrtl':
            raise RuntimeError('simulated evaluation crash')
        return original(manifest, cell, path, split, **kwargs)
    monkeypatch.setattr(runner, 'evaluate_saved', fail_at_four)
    with pytest.raises(RuntimeError, match='simulated'):
        runner.run_one(output, cell['id'], evaluation_budget=60)
    first = (folder / 'validation_0.json').read_bytes()
    saved = runner.load_session(cell, folder / 'checkpoint_4.rrtl')
    try:
        saved.run(4)
        expected_history = copy.deepcopy(saved.history)
        reference = saved.agent.encoder.transform([saved.state])
        expected_q = (saved.agent.predict_q(saved.state) if learner == 'corrected_common'
                      else saved.agent.model.predict(reference))
    finally:
        saved.environment.close()
    calls.clear()
    def counted(*args, **kwargs):
        calls.append(args[2].name)
        return original(*args, **kwargs)
    monkeypatch.setattr(runner, 'evaluate_saved', counted)
    runner.run_one(output, cell['id'], evaluation_budget=60)
    assert calls == ['checkpoint_4.rrtl', 'checkpoint_8.rrtl']
    assert (folder / 'validation_0.json').read_bytes() == first
    session = runner.load_session(cell, folder / 'final_checkpoint.rrtl')
    try:
        def no_times(value):
            if isinstance(value, dict):
                return {k: no_times(v) for k, v in value.items() if 'seconds' not in k}
            if isinstance(value, list):
                return [no_times(v) for v in value]
            return value
        assert no_times(session.history) == no_times(expected_history)
        q = (session.agent.predict_q(session.state) if learner == 'corrected_common'
             else session.agent.model.predict(reference))
        np.testing.assert_array_equal(q, expected_q)
        assert [r['transition_id'] for r in session.history] == list(range(8))
    finally:
        session.environment.close()


def test_setup_failure_does_not_evaluate_or_publish_completed(tmp_path, monkeypatch):
    output = tmp_path / 'campaign'
    manifest = runner.freeze(small_protocol(), output)
    cell = manifest['cells'][0]
    monkeypatch.setattr(runner, 'make_comparison_session', lambda *a, **k: (_ for _ in ()).throw(RuntimeError('setup')))
    monkeypatch.setattr(runner, 'evaluate_saved', lambda *a, **k: pytest.fail('evaluation after failed setup'))
    with pytest.raises(RuntimeError):
        runner.run_one(output, cell['id'], evaluation_budget=60)
    folder = output / cell['id']
    assert runner.read(folder / 'status.json')['state'] == 'failed'
    assert not list(folder.glob('validation*.json'))


@pytest.mark.atari
def test_budget_exhaustion_publishes_partial_checkpoint(tmp_path):
    protocol = small_protocol()
    protocol['budgets']['training_seconds_per_run'] = 1e-10
    output = tmp_path / 'campaign'
    manifest = runner.freeze(protocol, output)
    cell = manifest['cells'][0]
    status = runner.run_one(output, cell['id'], evaluation_budget=60)
    assert status['state'] == 'partial' and status['interactions'] < protocol['interactions']
    assert (output / cell['id'] / status['checkpoint']).exists()
    assert not (output / cell['id'] / 'final_checkpoint.rrtl').exists()


@pytest.mark.atari
@pytest.mark.parametrize('learner', ['bart', 'corrected_common'])
def test_exhausted_training_can_resume_with_recorded_extension(tmp_path, learner):
    protocol = small_protocol()
    protocol['budgets']['training_seconds_per_run'] = 1e-10
    output = tmp_path / 'campaign'
    manifest = runner.freeze(protocol, output)
    frozen = (output / 'protocol.json').read_bytes()
    cell = next(c for c in manifest['cells'] if c['pipeline'] == 'visual' and c['learner'] == learner)
    folder = output / cell['id']
    first = runner.run_one(output, cell['id'], evaluation_budget=60)
    assert first['state'] == 'partial' and first['interactions'] == 1
    history = runner.read(folder / 'train.json')
    validation = (folder / 'validation_0.json').read_bytes()
    again = runner.run_one(output, cell['id'], evaluation_budget=60)
    assert again['interactions'] == 1 and again['training_seconds'] == first['training_seconds']
    runner.extend_budgets(output, manifest, training_seconds=60, reason='Complete interrupted smoke')
    final = runner.run_one(output, cell['id'], evaluation_budget=60)
    assert final['state'] == 'completed' and final['interactions'] == 8
    assert final['training_seconds'] > first['training_seconds']
    assert final['evaluation_seconds'] >= first['evaluation_seconds']
    assert final['operational_budget_revision'] == 1 and 'reason' not in final
    assert runner.read(folder / 'train.json')[:1] == history
    assert [r['transition_id'] for r in runner.read(folder / 'train.json')] == list(range(8))
    assert (folder / 'validation_0.json').read_bytes() == validation
    assert (output / 'protocol.json').read_bytes() == frozen
    _, revisions = runner.operational_budgets(output, manifest)
    assert revisions[0]['consumed'][cell['id']]['training_seconds'] == first['training_seconds']


@pytest.mark.atari
@pytest.mark.ocatari
def test_global_evaluation_budget_cli_extension_completes_pending_runs(tmp_path):
    protocol = small_protocol()
    protocol['budgets']['evaluation_seconds_total'] = 1e-10
    source = tmp_path / 'protocol.json'
    source.write_text(json.dumps(protocol))
    output = tmp_path / 'campaign'
    def command(*args, success=True):
        result = subprocess.run([sys.executable, '-m', 'experiments.compare_pong', '--output', str(output), *args],
                                cwd=ROOT, capture_output=True, text=True, timeout=90)
        assert (result.returncode == 0) == success, result.stdout + result.stderr
    command('--stage', 'train', '--protocol', str(source))
    first = runner.read(output / 'campaign_status.json')
    assert first['evaluation_seconds'] > protocol['budgets']['evaluation_seconds_total']
    assert 'partial' in first['states'].values()
    frozen = (output / 'protocol.json').read_bytes()
    command('--stage', 'resume')
    assert runner.read(output / 'campaign_status.json') == first
    command('--stage', 'report-validation', '--evaluation-budget-seconds', '120', '--budget-reason', 'wrong stage', success=False)
    assert not (output / 'operational_budgets.json').exists()
    command('--stage', 'resume', '--evaluation-budget-seconds', '120', '--budget-reason', 'Finish validation after budget stop')
    final = runner.read(output / 'campaign_status.json')
    assert set(final['states'].values()) == {'completed'}
    assert final['evaluation_seconds'] > first['evaluation_seconds']
    assert final['operational_budget_revision'] == 1
    assert (output / 'protocol.json').read_bytes() == frozen
    assert not list(output.rglob('test.json'))
    report = runner.read(output / 'report_validation.json')
    assert all(r['evaluation_points'] is not None for r in report['runs'])
