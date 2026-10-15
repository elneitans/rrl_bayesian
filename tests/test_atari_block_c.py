"""C13–C15: slot contracts, common learner recovery, seed-level analysis and sealed test."""
import copy
from dataclasses import asdict
import json
import multiprocessing as mp
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from bayesian_rrtl.games.breakout import BreakoutRelationsV1,breakout_object_frame,BREAKOUT_SLOTS
from bayesian_rrtl.games.demonattack import DemonAttackRelationsV1,demonattack_object_frame,DEMON_SLOTS
from bayesian_rrtl.objects import DetectedObject,ObjectFrame
from bayesian_rrtl.online.environments import CommonGameEnvironment,GameEnvironmentConfig
from bayesian_rrtl.online.protocol import freeze,load,resolve,dry_run
from bayesian_rrtl.online.storage import binding,load_frontier,stream_rows
from bayesian_rrtl.evaluation import export_interpretability
from experiments.compare_pong_progressive import train_cell,evaluate_cell,consolidate
from experiments.atari_fact_controls import FactController
from experiments.pong_analysis_runtime import checked_read

ROOT=Path(__file__).resolve().parents[1]


def frame(names, *, x=70, ball=True):
    return ObjectFrame(tuple(DetectedObject(n,False) if n=='ball' and not ball else
        DetectedObject(n,True,(x+(8 if n=='ball' else 0),100.,8.,4.)) for n in names))


def keyfacts(facts):
    return {(f.name,f.obj1,f.obj2):f.value for f in facts}


def test_breakout_initial_absence_reappearance_and_reward_history():
    r=BreakoutRelationsV1()
    a=r.transform(frame(('player','ball')))
    d=keyfacts(a)
    assert ('x','ball_t','ball_t-1') not in d and ('present','ball_t-1',None) not in d
    b=r.transform(frame(('player','ball'),x=72),raw_reward=7.)
    assert keyfacts(b)[('x','ball_t','ball_t-1')]=='more'  # brick reward must not clear motion
    c=r.transform(frame(('player','ball'),ball=False))
    assert keyfacts(c)[('present','ball_t',None)] is False
    assert ('incontact','player_t','ball_t') not in keyfacts(c)
    d=r.transform(frame(('player','ball'),x=90))
    assert ('x','ball_t','ball_t-1') not in keyfacts(d)
    e=r.transform(frame(('player','ball'),x=91),invalidated=('ball',))
    assert ('x','ball_t','ball_t-1') not in keyfacts(e)
    assert keyfacts(e)[('x','player_t','player_t-1')]=='more'
    assert len(r.encoder().catalog)==11
    batch=r.encoder().transform([e,list(reversed(list(e)))])
    np.testing.assert_array_equal(batch.values[0],batch.values[1])
    saved=r.snapshot()
    restored=BreakoutRelationsV1()
    restored.restore(saved)
    assert restored.transform(frame(('ball','player'),x=92))==r.transform(frame(('player','ball'),x=92))


def objects(slots):
    return [SimpleNamespace(category=s.category,xywh=(20.,40.,4.,8.)) for s in slots]


@pytest.mark.parametrize('slots,adapter',[(BREAKOUT_SLOTS,breakout_object_frame),(DEMON_SLOTS,demonattack_object_frame)])
def test_detector_presence_invalid_bbox_and_unexpected_categories(slots,adapter):
    data=objects(slots)
    result=adapter(data)
    assert len(result.objects)==(2 if slots==BREAKOUT_SLOTS else 30)
    data[1]=SimpleNamespace(category='NoObject')
    assert next(o for o in adapter(data).objects if o.identity==slots[1].identity).present is False
    data[-1]=SimpleNamespace(category='Unexpected',xywh=(0,0,1,1))
    with pytest.raises(ValueError,match='category'):
        adapter(data)
    data=objects(slots)
    data[0].xywh=(float('nan'),0,1,1)
    with pytest.raises(ValueError,match='bbox'):
        adapter(data)
    with pytest.raises(ValueError,match='slots'):
        adapter(data[:-1])


def test_demon_slots_linear_catalog_no_unreliable_enemy_motion():
    r=DemonAttackRelationsV1()
    names=[s.identity for s in DEMON_SLOTS]
    initial=r.transform(frame(names))
    later=r.transform(frame(reversed(names),x=80),raw_reward=3.)
    assert len(r.encoder().catalog)==134
    assert len(r.encoder().catalog)<5*len(names)
    facts=keyfacts(later)
    assert facts[('x','player_t','player_t-1')]=='more'
    assert not any(k[2] and 'enemy' in k[1] and '_t-1' in k[2] for k in facts)
    # Uncertifiable slot reuse never turns into velocity of a physical enemy/projectile.
    disappeared=ObjectFrame(tuple(DetectedObject(n,False) if 'enemy' in n else
                                 DetectedObject(n,True,(0,0,4,8)) for n in names))
    r.transform(disappeared)
    new=r.transform(frame(names,x=120))
    assert ('x','enemy_0_t','enemy_0_t-1') not in keyfacts(new)
    assert keyfacts(new)[('present','enemy_0_t-1',None)] is False
    assert initial


class FakeBackend:
    def __init__(self,config):
        self.config=config
        self.spec=config.spec()
        self.actions=self.spec.actions
        self.steps=0
        self.seed=None
        self.current_frame=None
    def make_frame(self):
        names=('player','ball') if self.config.game=='Breakout' else tuple(s.identity for s in self.spec.slots)
        return frame(names,x=30+self.steps)
    def reset(self, *, seed):
        self.steps,self.seed=0,seed
        self.current_frame=self.make_frame()
        return self.current_frame
    def step(self,action):
        self.steps+=1
        self.current_frame=self.make_frame()
        raw=5. if self.steps==3 else 0.
        return SimpleNamespace(frame=self.current_frame,raw_reward=raw,terminated=self.steps==9,truncated=False)
    def lives(self):
        return 3-int(self.steps>=4)
    def manifest(self):
        return {'fake_game':self.config.game,'horizon':self.config.horizon}
    def snapshot(self):
        return {'steps':self.steps,'seed':self.seed}
    def restore(self,saved):
        self.steps,self.seed=saved['steps'],saved['seed']
        self.current_frame=self.make_frame()
    def close(self):
        pass


class FakeFactory:
    def __init__(self,config):
        self.config=config
    def __call__(self):
        return CommonGameEnvironment(self.config,backend=FakeBackend(self.config))
    def with_horizon(self,horizon):
        return FakeFactory(GameEnvironmentConfig(**{**asdict(self.config),'horizon':horizon}))


def protocol(game='Breakout'):
    p=json.loads((ROOT/'configs/atari_comparison_smoke.json').read_text())
    p['games']={game:p['games'][game]}
    p['cells']=[c for c in p['cells'] if c['game']==game]
    # Discardable synthetic seed groups, not the actual frozen reserved test lists.
    p['games'][game]['evaluation']['reserved_test']['seeds']=[991901]
    p['games'][game]['evaluation']['reserved_test']['horizon']=16
    return p


def test_life_loss_does_not_end_or_clear_paddle_history():
    config=GameEnvironmentConfig(**protocol()['games']['Breakout']['environment'])
    env=FakeFactory(config)()
    env.reset(seed=42)
    for _ in range(3):
        facts,*_=env.step(0)
    assert keyfacts(facts)[('x','ball_t','ball_t-1')]=='more'
    facts,reward,raw,terminal,truncated=env.step(0)
    assert not terminal and not truncated and env.last_auxiliary['life_lost']
    assert ('x','ball_t','ball_t-1') not in keyfacts(facts)
    assert keyfacts(facts)[('x','player_t','player_t-1')]=='more'


@pytest.mark.parametrize('game',['Breakout','DemonAttack'])
def test_three_learners_common_runner_effective_fits_resume_analysis(tmp_path,game):
    p=protocol(game)
    one,two=tmp_path/'full',tmp_path/'resume'
    m=freeze(p,one)
    freeze(p,two)
    factory=FakeFactory(GameEnvironmentConfig(**p['games'][game]['environment']))
    for index,cell in enumerate(m['cells']):
        train_cell(one,index,factory=factory)
        train_cell(two,index,factory=factory,decision_limit=19)
        s=train_cell(two,index,factory=factory)
        assert s['state']=='completed'
        if cell['learner']!='corrected_common':
            assert s['metrics']['learner']['fits']>=2
        def read(folder):
            status=checked_read(folder/cell['id']/'status.json')
            data,_=load_frontier(folder/cell['id'],status,binding(load(folder),cell))
            return data,list(stream_rows(folder/cell['id'],status['log_head']))
        a,x=read(one)
        b,y=read(two)
        assert [{k:v for k,v in r.items() if k!='fit'} for r in x]==[{k:v for k,v in r.items() if k!='fit'} for r in y]
        assert all('points_won' not in r for r in y)
        evaluate_cell(two,index,factory=factory)
        assert read(two)[1]==y
        agent=b['session']['collector']['agent']
        exported=export_interpretability(agent,cell['learner'],[[],b['session']['collector']['state']])
        assert len(exported['reference_q'])==2
    report=consolidate(two)
    assert report['completed'] and len(report['analysis']['seed_summaries'])==3
    assert len(report['analysis']['paired_contrasts'])==6
    assert all(len(r['curve'])==3 and r['normalized_auc_decisions'] is not None for r in report['analysis']['runs'])
    assert all('points_won' not in e for r in report['cells'] for v in r['evaluations'] for e in v['report']['episodes'])


def test_protocol_exact_cells_no_implicit_cartesian_or_visual_expansion(tmp_path):
    p=protocol()
    r=resolve(p)
    assert len(r['cells'])==3 and all(c['id'].startswith('Breakout/ocatari/') for c in r['cells'])
    assert len(r['cells'][0]['learning']['actions'])==4
    bad=copy.deepcopy(p)
    bad['cells'][0]['pipeline']='visual'
    with pytest.raises(ValueError,match='Pong'):
        resolve(bad)
    bad=copy.deepcopy(p)
    bad['cells'].append(bad['cells'][0])
    with pytest.raises(ValueError,match='Duplicate'):
        resolve(bad)
    pending=json.loads((ROOT/'configs/atari_comparison_final_pending.json').read_text())
    d=dry_run(pending)
    assert d['n_trainings']==45 and d['pending_authorizations'] and d['estimates']==[]
    assert all(n['decisions'] is None for n in d['workload'])


def test_fact_controls_only_facts_and_deterministic_missing_rules():
    config=GameEnvironmentConfig(**protocol()['games']['Breakout']['environment'])
    env=FakeFactory(config)()
    rng=np.random.default_rng(42)
    control=FactController('Breakout')
    facts=BreakoutRelationsV1().transform(frame(('player','ball'),ball=False))
    assert control.action(facts,env,rng)==1
    facts=BreakoutRelationsV1().transform(frame(('player','ball')))
    assert control.action(facts,env,rng)==2
    assert control.action([],env,rng)==0


def test_final_test_requires_seal_and_is_resumable_without_training(tmp_path,monkeypatch):
    from bayesian_rrtl.online import final_test
    from bayesian_rrtl.online.final_test import prepare,authorize,evaluate
    p=protocol()
    p['games']['Breakout']['evaluation']['reserved_test']['seeds']=[991901,991902,991903]
    folder=tmp_path/'seal'
    m=freeze(p,folder)
    factory=FakeFactory(GameEnvironmentConfig(**p['games']['Breakout']['environment']))
    with pytest.raises(ValueError,match='complete'):
        prepare(folder,tmp_path/'selection.json')
    for i in range(3):
        train_cell(folder,i,factory=factory)
        evaluate_cell(folder,i,factory=factory)
    selection=tmp_path/'selection.json'
    packet=prepare(folder,selection)
    with pytest.raises(ValueError,match='confirm'):
        authorize(folder,selection,reason='unit test')
    packet['selection_decisions']='Synthetic development-only acceptance; no real test seeds'
    packet['test_seeds_untouched_attestation']=True
    selection.write_text(json.dumps(packet))
    authorize(folder,selection,confirmed=True,reason='discardable synthetic unit')
    cell=m['cells'][0]
    status=(folder/cell['id']/'status.json').read_bytes()
    with pytest.raises(ValueError,match='confirm'):
        evaluate(folder,0,factory=factory)
    # Preserve a complete first episode and a partial second, then simulate an
    # interruption before the third. Retrying must neither replay nor lose steps.
    original_episode=final_test.episode
    original_step=FakeBackend.step
    calls=[]
    steps={}
    def counted_step(self,action):
        steps[self.seed]=steps.get(self.seed,0)+1
        return original_step(self,action)
    def interrupted_episode(*args,**kwargs):
        calls.append(kwargs['seed'])
        if len(calls)==2:
            kwargs['decision_limit']=3
        if len(calls)==3:
            raise OSError('Simulated interruption after 1 complete and 1 partial')
        return original_episode(*args,**kwargs)
    monkeypatch.setattr(FakeBackend,'step',counted_step)
    monkeypatch.setattr(final_test,'episode',interrupted_episode)
    with pytest.raises(OSError,match='Simulated'):
        evaluate(folder,0,confirmed=True,factory=factory)
    first_unit=folder/cell['id']/'evaluations'/'reserved_test'/'64'/'991901'/'state.json'
    complete_bytes=first_unit.read_bytes()
    assert steps=={991901:9,991902:3}
    monkeypatch.setattr(final_test,'episode',original_episode)
    first=evaluate(folder,0,confirmed=True,factory=factory)
    second=evaluate(folder,0,confirmed=True,factory=factory)
    assert first==second and first['state']=='completed'
    assert first_unit.read_bytes()==complete_bytes
    assert steps=={991901:9,991902:9,991903:9}
    assert (folder/cell['id']/'status.json').read_bytes()==status
    assert consolidate(folder)['reserved_test_executed'] is True
    assert consolidate(folder)['reserved_test_executed_by_report'] is False


@pytest.mark.parametrize('game',['Breakout','DemonAttack'])
def test_life_terminal_and_truncation_targets_use_common_bootstrap(game):
    from bayesian_rrtl.online.learners import LearningConfig,OnlineBatchAgent
    from bayesian_rrtl.targets import frozen_q_targets
    p=protocol(game)
    cfg=GameEnvironmentConfig(**p['games'][game]['environment'])
    env=FakeFactory(cfg)()
    state=env.reset(seed=7)
    agent=OnlineBatchAgent(LearningConfig(actions=env.actions,seed=42),env.encoder(),
                           learner='cart',estimator={'max_depth':2,'min_child':1})
    for step in range(9):
        following,reward,raw,term,trunc=env.step(0)
        agent.observe(state,0,reward,following,term,trunc,raw_reward=raw,episode_id=0,step_id=step)
        state=following
    class Ten:
        def predict(self,batch):
            return np.full((len(batch),len(env.actions)),10.)
    transitions=agent.replay.transitions
    y=frozen_q_targets(transitions,Ten(),env.encoder(),gamma=.99,bootstrap_on_truncation=True)
    assert y[3]==9.9  # life lost, still bootstraps
    assert y[-1]==0.  # actual terminal removes bootstrap
    from dataclasses import replace
    t=replace(transitions[3],truncated=True)
    assert frozen_q_targets([t],Ten(),env.encoder(),gamma=.99,bootstrap_on_truncation=True)[0]==9.9
    assert frozen_q_targets([t],Ten(),env.encoder(),gamma=.99,bootstrap_on_truncation=False)[0]==0.


def test_codec_rejects_mutable_fields_and_unknown_slot_categories():
    from bayesian_rrtl.detector_codec import DetectorCodec,BASE_FIELDS
    from bayesian_rrtl.game_registry import SlotSpec
    codec=DetectorCodec('Breakout','ocatari.ram.breakout',(SlotSpec(0,'player','Player'),))
    class Player:
        category='Player'
        def __init__(self):
            for name in BASE_FIELDS:
                setattr(self,name,None)
    p=Player()
    assert codec.encode([p])[0]['category']=='Player'
    p.unexpected=1
    with pytest.raises(ValueError,match='mutable'):
        codec.encode([p])
    del p.unexpected
    p.category='Unknown'
    with pytest.raises(ValueError,match='category'):
        codec.encode([p])


def test_missing_common_auc_and_failed_runs_are_not_imputed_to_zero(tmp_path):
    p=protocol()
    folder=tmp_path/'partial-curve'
    freeze(p,folder)
    factory=FakeFactory(GameEnvironmentConfig(**p['games']['Breakout']['environment']))
    train_cell(folder,0,factory=factory,decision_limit=19)
    evaluate_cell(folder,0,point=0,group='intermediate',factory=factory)
    r=consolidate(folder)
    assert not r['completed'] and len(r['analysis']['runs'])==3
    assert all(run['normalized_auc_decisions'] is None for run in r['analysis']['runs'])
    assert all(group['mean'] is None for group in r['analysis']['seed_summaries'])
    assert any(run['state']=='pending' for run in r['analysis']['runs'])


def test_sealed_test_report_does_not_fill_missing_episodes(tmp_path):
    from bayesian_rrtl.online.final_test import prepare,authorize,report
    p=protocol()
    folder=tmp_path/'test-report'
    freeze(p,folder)
    factory=FakeFactory(GameEnvironmentConfig(**p['games']['Breakout']['environment']))
    for i in range(3):
        train_cell(folder,i,factory=factory)
        evaluate_cell(folder,i,factory=factory)
    path=tmp_path/'selection.json'
    packet=prepare(folder,path)
    packet['selection_decisions']='Synthetic fixture only'
    packet['test_seeds_untouched_attestation']=True
    path.write_text(json.dumps(packet))
    authorize(folder,path,confirmed=True,reason='Synthetic fixture')
    result=report(folder)
    assert result['state']=='partial'
    assert all(v['report'] is None for v in result['runs'])
    assert not (folder/'final_test'/'usage.json').exists()


def paused_reserved_test_worker(folder,ready,release):
    """A real competing process holds the episode lock inside evaluation."""
    from bayesian_rrtl.online.final_test import evaluate
    config=GameEnvironmentConfig(**load(folder)['protocol']['games']['Breakout']['environment'])
    original=FakeBackend.step
    notified=False
    def paused_step(self,action):
        nonlocal notified
        if not notified:
            notified=True
            ready.send('evaluating')
            if not release.wait(30):
                raise TimeoutError('Concurrency regression coordinator did not release evaluator')
        return original(self,action)
    FakeBackend.step=paused_step
    try:
        evaluate(folder,0,confirmed=True,factory=FakeFactory(config))
    finally:
        FakeBackend.step=original
        ready.close()


def test_sealed_test_report_handles_active_evaluator_without_reading_progress(tmp_path,monkeypatch):
    from bayesian_rrtl.online.final_test import prepare,authorize,evaluate,report
    from experiments.pong_analysis_runtime import WorkUnit
    p=protocol()
    folder=tmp_path/'active-test-report'
    manifest=freeze(p,folder)
    factory=FakeFactory(GameEnvironmentConfig(**p['games']['Breakout']['environment']))
    for i in range(3):
        train_cell(folder,i,factory=factory)
        evaluate_cell(folder,i,factory=factory)
    selection=tmp_path/'selection.json'
    packet=prepare(folder,selection)
    packet['selection_decisions']='Synthetic concurrency regression only'
    packet['test_seeds_untouched_attestation']=True
    selection.write_text(json.dumps(packet))
    authorize(folder,selection,confirmed=True,reason='Synthetic fixture')
    cell=manifest['cells'][0]
    status=folder/cell['id']/'status.json'
    training_before=status.read_bytes()
    unit=folder/cell['id']/'evaluations'/'reserved_test'/'64'/'991901'
    context=mp.get_context('spawn')
    ready,child_ready=context.Pipe(duplex=False)
    release=context.Event()
    process=context.Process(target=paused_reserved_test_worker,args=(folder,child_ready,release))
    process.start()
    child_ready.close()
    try:
        assert ready.poll(30) and ready.recv()=='evaluating'
        before={path.name:path.read_bytes() for path in unit.rglob('*') if path.is_file()}
        original=WorkUnit.load
        def guarded_load(self):
            if self.folder==unit:
                raise AssertionError('Report read progress while evaluation held the episode lock')
            return original(self)
        with monkeypatch.context() as patch:
            patch.setattr(WorkUnit,'load',guarded_load)
            result=report(folder)
            from experiments.compare_pong_progressive import main
            assert main(['--stage','report-test','--campaign',str(folder)])==3
        assert result['state']=='partial'
        run=result['runs'][0]
        assert run['state']=='partial' and run['report'] is None
        assert run['pending_evaluation_seeds']==[991901]
        assert run['pending_evaluations']==[{'seed':991901,'reason':'evaluation_running'}]
        assert checked_read(folder/'final_test'/'report.json')==result
        assert {path.name:path.read_bytes() for path in unit.rglob('*') if path.is_file()}==before
        assert status.read_bytes()==training_before
    finally:
        release.set()
        process.join(30)
        if process.is_alive():
            process.kill()
            process.join(5)
        ready.close()
    assert process.exitcode==0
    for i in (1,2):
        evaluate(folder,i,confirmed=True,factory=factory)
    assert report(folder)['state']=='completed'
    assert status.read_bytes()==training_before
    # Unrelated runtime failures and corrupt results must remain visible.
    def broken_load(self):
        raise RuntimeError('Unexpected read failure')
    with monkeypatch.context() as patch:
        patch.setattr(WorkUnit,'load',broken_load)
        with pytest.raises(RuntimeError,match='Unexpected read failure'):
            report(folder)
    state=unit/'state.json'
    state.write_text('{}')
    with pytest.raises(ValueError,match='Corrupt analysis JSON'):
        report(folder)
