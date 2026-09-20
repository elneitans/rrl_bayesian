from argparse import Namespace
import subprocess

import pytest

from experiments.cluster_launch import ROOT, campaign_lock, command_for
from experiments import cluster_launch


def options(**kwargs):
    return Namespace(**dict(dict(output='results/example', stage='train', kind='smoke',
        training_budget_seconds=None, evaluation_budget_seconds=None, budget_reason=None), **kwargs))


def test_train_uses_single_coordinator_and_no_test():
    command = command_for(options())
    assert command[command.index('--stage') + 1] == 'train'
    assert command[-1].endswith('configs/comparison_pong_smoke.json')
    assert '--worker' not in command and '--confirm-test' not in command


def test_resume_keeps_frozen_protocol_and_explicit_budget():
    command = command_for(options(stage='resume', kind=None, training_budget_seconds=9000,
                                  budget_reason='Authorized additional runtime'))
    assert '--protocol' not in command
    assert '--training-budget-seconds' in command
    with pytest.raises(ValueError):
        command_for(options(stage='resume', kind=None, training_budget_seconds=9000))
    with pytest.raises(ValueError):
        command_for(options(stage='resume'))
    with pytest.raises(ValueError):
        command_for(options(training_budget_seconds=9000))


def test_campaign_lock_rejects_second_owner_and_releases(tmp_path):
    path = tmp_path / 'campaign'
    with campaign_lock(path):
        with pytest.raises(RuntimeError, match='coordinator'):
            with campaign_lock(path):
                pass
    with campaign_lock(path):
        assert not path.exists()  # freeze must create a fresh directory itself


@pytest.mark.parametrize('state,expected', [('completed', 0), ('partial', 3), ('failed', 3)])
def test_exit_status_reflects_campaign_not_just_subprocess(tmp_path, monkeypatch, state, expected):
    import json
    output = tmp_path / 'campaign'
    output.mkdir()
    (output / 'campaign_status.json').write_text(json.dumps({'states': {'cell': state}}))
    calls = []
    monkeypatch.setattr(cluster_launch.subprocess, 'run', lambda *a, **kw: calls.append((a, kw)))
    monkeypatch.setattr(cluster_launch.sys, 'argv', ['cluster_launch', '--stage', 'resume', '--output', str(output)])
    assert cluster_launch.main() == expected
    assert len(calls) == 1 and calls[0][1]['check'] is True


@pytest.mark.parametrize('script', sorted((ROOT / 'slurm').glob('*.sbatch')) + [ROOT / 'slurm/common.sh'])
def test_shell_syntax_and_no_gpu_or_array(script):
    subprocess.run(['bash', '-n', str(script)], check=True)
    text = script.read_text()
    assert '#SBATCH --gres' not in text and '#SBATCH --array' not in text
    assert '--confirm-test' not in text
