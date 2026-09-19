"""Step 5 protocol and budget supervision without emulation."""
import json
from pathlib import Path
import subprocess

import pytest

from experiments.validate_ocatari_pong import run_budgeted

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('name,steps,m,samples,capacity,batch,interval,limit', [
    ('smoke', 2048, 5, 20, 5000, 64, 256, 1000),
    ('pilot', 20480, 20, 50, 50000, 128, 512, 27000)])
def test_full_predeclared_configuration(name, steps, m, samples, capacity, batch, interval, limit):
    from bayesian_rrtl.atari import AtariConfig
    from bayesian_rrtl.q_config import BayesianQConfig
    from train_atari import validate_pair
    raw = json.loads((ROOT / f'configs/bayesian_pong_ocatari_{name}.json').read_text())
    env, agent = AtariConfig(**raw['environment']), BayesianQConfig(**raw['agent'])
    assert set(raw['agent']) == set(agent.to_dict())
    validate_pair(env, agent)
    assert raw['interactions'] == steps and agent.m == m
    assert agent.burn_in == agent.draws == samples
    assert agent.replay_capacity == capacity and agent.batch_size_per_action == batch
    assert agent.warmup_interactions == agent.fit_interval == interval
    assert env.max_episode_steps == limit
    assert agent.k == 2 and agent.min_child == 2 and agent.max_depth == 4
    assert agent.epsilon_decay_steps == 50000 and not agent.resample_repeated_actions
    assert agent.test_seeds == (410001, 410002) and agent.validation_seeds == (310011, 310012)
    assert raw['training_seeds'] == ([41] if name == 'smoke' else [41, 42, 43])
    assert raw['diagnostics']['chains'] == 4 and raw['diagnostics']['burn_in'] == 500
    assert raw['diagnostics']['draws'] == 1000


def test_budget_timeout_preserves_worker_artifacts(tmp_path, monkeypatch):
    def timeout(*args, **kwargs):
        (tmp_path / 'progress_checkpoint.rrtl').write_bytes(b'completed block')
        raise subprocess.TimeoutExpired(args[0], kwargs['timeout'])
    monkeypatch.setattr(subprocess, 'run', timeout)
    result = run_budgeted(['worker'], tmp_path / 'log', 600)
    assert result['status'] == 'not_run' and result['budget_seconds'] == 600
    assert (tmp_path / 'progress_checkpoint.rrtl').read_bytes() == b'completed block'


def test_worker_failure_is_not_reported_as_success(tmp_path, monkeypatch):
    monkeypatch.setattr(subprocess, 'run', lambda *a, **kw: subprocess.CompletedProcess(a[0], 1))
    assert run_budgeted(['worker'], tmp_path / 'log', 600)['status'] == 'failed'
