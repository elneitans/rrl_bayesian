from dataclasses import asdict
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from bayesian_rrtl.agent import BayesianQAgent
from bayesian_rrtl.encoding import RelationalEncoder, RelationKey
from bayesian_rrtl.q_config import BayesianQConfig
from preprocessing import Fact
from experiments import audit_pong_readiness as audit
from test_comparison_protocol import small_protocol


def fitted():
    encoder = RelationalEncoder([RelationKey('comparative', 'x', 'a', 'b')])
    config = BayesianQConfig(actions=('a',), m=1, burn_in=1, draws=2, fit_interval=1)
    agent = BayesianQAgent(config, encoder)
    for i in range(12):
        state = [Fact('comparative', ('less' if i < 9 else 'more'), 'x', 'a', 'b')]
        agent.observe(state, 0, float(i % 2), state, False, False, raw_reward=0., episode_id=0, step_id=i)
    agent.fit_if_due()
    return agent


def test_reference_selection_uses_replay_frequencies_only():
    agent = fitted()
    reference, selected = audit.reference_states(agent, count=2)
    assert [item['frequency'] for item in selected] == [9, 3]
    assert [item['group'] for item in selected] == ['frequent', 'rare']
    assert reference.values.shape == (2, 1)
    assert asdict(agent.config)['test_seeds'] == (200001, 200002)


def test_audit_kernels_share_fixed_data_prior_lengths_and_seed(tmp_path, monkeypatch):
    agent = fitted()
    reference, selected = audit.reference_states(agent)
    payload = {'config': agent.config, 'fit': agent.last_fit, 'calibrations': agent.calibrations,
               'reference': reference, 'reference_selection': selected, 'checkpoint_sha256': 'fixed'}
    path = tmp_path / 'fixed.pkl'
    path.write_bytes(pickle.dumps(payload))
    calls = []
    class Model:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
        def fit(self, batch, targets, rng, *, burn_in, draws):
            calls.append((self.kwargs, batch.values.copy(), targets.copy(), rng.bit_generator.state, burn_in, draws))
            return SimpleNamespace(predict_latent_draws=lambda batch: np.zeros((1000, len(batch))),
                sigma2=np.ones(1000), trace=[SimpleNamespace(total_leaves=1, max_depth=0)]*1500,
                diagnostics=lambda: {'elapsed_seconds': .1})
    monkeypatch.setattr(audit, 'BARTRegressor', Model)
    monkeypatch.setattr(audit, 'posterior_diagnostics', lambda *args: {
        'sigma2': {'status': 'constant', 'rhat': None, 'ess_bulk': None, 'ess_tail': None}})
    for kernel in ('grow_prune', 'revision'):
        audit.diagnose_action(path, 0, kernel, tmp_path / kernel)
    assert len(calls) == 8
    for left, right in zip(calls[:4], calls[4:]):
        assert left[0]['kernel'] == 'grow_prune' and right[0]['kernel'] == 'revision'
        assert left[0]['noise_prior'] == right[0]['noise_prior'] == agent.calibrations[0].prior
        assert left[0]['scale'] == right[0]['scale'] == agent.config.scale
        np.testing.assert_array_equal(left[1], right[1])
        np.testing.assert_array_equal(left[2], right[2])
        assert left[3:] == right[3:]
        assert left[-2:] == (500, 1000)
    report = audit.read(tmp_path / 'revision/report.json')
    assert not report['all_observables_pass'] and report['constant_observables'] == ['sigma2']
    assert len(list((tmp_path / 'revision').glob('chain_*.npz'))) == 4


def test_readiness_budget_rejects_unbounded_requests(tmp_path):
    for value in (0, -1, 1201):
        with pytest.raises(ValueError, match='budget'):
            audit.audit(tmp_path, tmp_path / 'out', tmp_path / 'pilot', budget_seconds=value)


def test_pilot_cost_includes_all_validation_episodes_and_horizon():
    smoke = small_protocol()
    pilot = small_protocol()
    pilot.update(training_seeds=[81,82,83], interactions=20480, curve_checkpoints=[0,5120,10240,15360,20480],
                 validation_seeds=list(range(330001,330011)), evaluation_episodes=10, evaluation_horizon=27000)
    summaries = []
    for pipeline in ('visual', 'ocatari'):
        for learner in ('corrected_common', 'bart'):
            summaries.append(({'pipeline': pipeline, 'learner': learner},
                              {'timing': {'training_seconds': 1., 'save_seconds': .1}},
                              [{'seconds': 6., 'report': {'episodes': [{'steps': 6}]}}]))
    result = audit.estimate_cost({'protocol': smoke}, summaries, pilot)
    assert result['training_runs'] == 12 and result['validation_episodes'] == 600
    assert sum(c['validation_episodes'] for c in result['cells']) == 600
    assert result['horizon_seconds'] > result['central_seconds'] > 0
