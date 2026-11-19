from dataclasses import asdict
import copy

import pytest

from bayesian_rrtl.stable_revision import StableRevisionAgent,variant_config
from bayesian_rrtl.tree_revision import RevisionConfig
from experiments import diagnose_rrtl_improvements as runner
from experiments.diagnose_rrtl_revision import save_fork,restore_fork
from test_revision_runner import Environment,session


def fixture():
    a=session();cfg=variant_config('stable_td',target_interval=7)
    revision=RevisionConfig(seed=1072715)
    a.agent=StableRevisionAgent.fork(a.agent,revision,cfg)
    identity={'parent_counter':0,'campaign':'synthetic_improvements','seed':4,'mode':'stable_td'}
    manifest={'config':{'revision':{**asdict(revision),'seed':1072711},'variants':{'stable_td':asdict(cfg)}}}
    return a,identity,manifest


def test_improvement_checkpoint_resumes_td_targets_rng_and_environment(tmp_path):
    a,identity,m=fixture()
    initial=save_fork(tmp_path,identity,a,None,schema=runner.FORK_SCHEMA)
    for _ in range(19):a.step()
    saved=save_fork(tmp_path,identity,a,initial,schema=runner.FORK_SCHEMA)
    b=runner.restore(tmp_path,saved,identity,Environment(),m)
    for _ in range(51):
        ar,br=a.step(),b.step()
        assert {k:v for k,v in ar.items() if k!='fit'}=={k:v for k,v in br.items() if k!='fit'}
    assert a.agent.tree_export()==b.agent.tree_export()
    assert a.agent.target_counter==b.agent.target_counter==70
    assert a.agent.target_syncs==b.agent.target_syncs


def test_original_runner_rejects_new_schema_and_settings_are_bound(tmp_path):
    a,identity,m=fixture();saved=save_fork(tmp_path,identity,a,None,schema=runner.FORK_SCHEMA)
    with pytest.raises(ValueError,match='identity'):restore_fork(tmp_path,saved,identity,Environment())
    changed=copy.deepcopy(m);changed['config']['variants']['stable_td']['eta_q']=.01
    with pytest.raises(ValueError,match='settings differ'):runner.restore(tmp_path,saved,identity,Environment(),changed)


def test_corrupt_target_forest_or_frontier_is_rejected():
    a,_,_=fixture();a.agent.target_roots=a.agent.root_nodes
    with pytest.raises(ValueError,match='aliases'):a.agent.validate_revision()
    b,_,_=fixture();b.agent.target_counter=1
    with pytest.raises(ValueError,match='frontier'):b.agent.validate_revision()
