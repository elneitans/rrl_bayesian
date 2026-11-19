import copy
from dataclasses import asdict
import gzip
import json
import pickle

import numpy as np
import pytest

from bayesian_rrtl.online.learners import LearningConfig
from bayesian_rrtl.online.session import OnlineSession
from bayesian_rrtl.tree_revision import RevisionAgent, RevisionConfig
from experiments.diagnose_rrtl_revision import restore_fork, save_fork
from experiments.pong_analysis_runtime import checked_read
from test_tree_revision import base, state


class Environment:
    actions = ('NOOP','FIRE','RIGHT','LEFT')
    last_relation_seconds = 0.
    def __init__(self): self.steps = 0; self.done = True; self.last_auxiliary = {'life_lost':False}
    def encoder(self): return base().encoder
    def reset(self, *, seed): self.steps=0; self.done=False; return state()
    def step(self, action):
        self.steps+=1; self.done=self.steps==7
        return state('less' if self.steps%2 else 'more'),float(action==2),float(action==2),self.done,False
    def snapshot(self): return {'steps':self.steps,'done':self.done}
    def restore(self, saved): self.steps=saved['steps']; self.done=saved['done']
    def runtime_manifest(self): return {'kind':'diagnostic_test_driver'}
    def lives(self): return 5
    def close(self): pass


def session():
    learning = LearningConfig(actions=Environment.actions, seed=4)
    cell = {'game':'Breakout','learner':'corrected_common','learning':asdict(learning),
            'estimator':{'eta_q':.2,'max_depth':3,'min_sample_size':6,'significance_level':.05,'split_test':'anova'}}
    result = OnlineSession(cell,Environment())
    result.agent = RevisionAgent.fork(result.agent,RevisionConfig(mode='prune_regrow',cooldown=4))
    return result


def test_fork_roundtrip_resumes_exact_td_rng_and_environment(tmp_path):
    a=session();identity={'parent_counter':0,'campaign':'test','mode':'prune_regrow'}
    initial=save_fork(tmp_path,identity,a,None)
    for _ in range(19): a.step()
    saved=save_fork(tmp_path,identity,a,initial)
    b=restore_fork(tmp_path,saved,identity,Environment())
    for _ in range(51):
        ar,br=a.step(),b.step()
        assert {k:v for k,v in ar.items() if k!='fit'} == {k:v for k,v in br.items() if k!='fit'}
    assert a.agent.tree_export()==b.agent.tree_export()
    final=save_fork(tmp_path,identity,a,saved)
    assert final['counter']==70
    assert sum(ref['end']-ref['start']+1 for ref in final['traces'])==70


def test_corrupt_trace_is_rejected_before_restore(tmp_path):
    a=session();identity={'parent_counter':0,'campaign':'test'}
    initial=save_fork(tmp_path,identity,a,None)
    for _ in range(4): a.step()
    saved=save_fork(tmp_path,identity,a,initial)
    path=tmp_path/saved['traces'][0]['path']; path.write_bytes(path.read_bytes()+b'corrupt')
    with pytest.raises(ValueError,match='trace chain'): restore_fork(tmp_path,saved,identity,Environment())


def test_fork_identity_change_is_rejected(tmp_path):
    a=session();identity={'parent_counter':0,'campaign':'test'}
    saved=save_fork(tmp_path,identity,a,None)
    with pytest.raises(ValueError,match='identity'): restore_fork(tmp_path,saved,{**identity,'campaign':'changed'},Environment())


def test_frozen_rollout_preserves_model_encoder_and_rng(monkeypatch):
    from bayesian_rrtl import revision_diagnostics as d
    a=session(); before=pickle.dumps(a.agent,protocol=4)
    start={'id':0,'physical_hash':'synthetic','state':state(),'environment':{'steps':0,'done':False}}
    def verify(env, saved): env.restore(saved['environment']); return saved['state']
    monkeypatch.setattr(d,'verify_start',verify)
    result=d.rollout(a.environment,a.agent,start,epsilon=.1,horizon=20,seed=1)
    assert result['steps']==7 and result['terminated']
    assert pickle.dumps(a.agent,protocol=4)==before


def test_distinct_start_banks_require_real_ram_and_relational_diversity():
    from types import SimpleNamespace
    from bayesian_rrtl import revision_diagnostics as d
    class StartsEnvironment(Environment):
        def __init__(self):
            super().__init__()
            self.offset=0
            self.backend=SimpleNamespace(env=SimpleNamespace(_env=SimpleNamespace(unwrapped=
                SimpleNamespace(ale=SimpleNamespace(getRAM=lambda: np.array([self.steps,self.offset],dtype=np.uint8))))))
            self.relations=SimpleNamespace(snapshot=lambda:{'step':self.steps})
        def reset(self, *, seed):
            self.steps=0;self.offset=seed%251;self.done=False;return state()
        def step(self, action):
            self.steps+=1
            labels=['less','same','more']
            return state(labels[(self.steps//16)%3],labels[(self.steps//32)%3]),0.,0.,False,False
        def snapshot(self):return {'steps':self.steps,'offset':self.offset,'done':False,
            'backend':{'ale_state':(self.steps,self.offset),'objects':[]},
            'relations':self.relations.snapshot()}
        def restore(self,saved):self.steps=saved['steps'];self.offset=saved['offset'];self.done=False
    env=StartsEnvironment()
    bank=d.make_start_bank(env,count=6,seed=8)
    assert bank['unique_ram']==6 and bank['unique_encoded']>=4
    validation=d.make_start_bank(env,count=6,seed=10,excluded_ram=[s['ram_sha256'] for s in bank['starts']])
    assert not {s['ram_sha256'] for s in bank['starts']} & {s['ram_sha256'] for s in validation['starts']}
    for start in bank['starts']:assert d.verify_start(env,start)==start['state']
    bad=copy.deepcopy(bank['starts'][0]);bad['environment']['steps']+=1
    with pytest.raises(ValueError,match='continuity'):d.verify_start(env,bad)
