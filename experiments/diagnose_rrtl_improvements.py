"""Prospective stable-TD/guarded-revision campaign. Original runs are read-only."""
import argparse
import copy
from dataclasses import asdict
import json
from pathlib import Path
import pickle

import numpy as np

from bayesian_rrtl.online.protocol import runtime, source_hashes
from bayesian_rrtl.online.session import OnlineSession
from bayesian_rrtl.online.storage import stream_rows
from bayesian_rrtl.baseline_adapter import normalize_facts
from bayesian_rrtl.revision_diagnostics import make_start_bank, verify_start, physical_hash, ram_hash, rollout, validation_probes
from bayesian_rrtl.stable_revision import VARIANTS, ImprovementConfig, StableRevisionAgent, return_gate, variant_config
from bayesian_rrtl.tree_revision import RevisionConfig
from experiments import diagnose_rrtl_revision as common
from experiments.compare_pong_progressive import StopRequest, default_factory
from experiments.pong_analysis_runtime import checked_read, checked_write, digest, unit_lock

SCHEMA='rrtl_improvement_campaign_v1'
FORK_SCHEMA='rrtl_improvement_fork_v1'


def load_campaign(output):
    output=Path(output);m=checked_read(output/'manifest.json')
    if m['schema']!=SCHEMA or m['sources']!=source_hashes() or m['runtime']!=runtime('corrected_common'):
        raise ValueError('Improvement source/runtime binding mismatch')
    validate_config(m['config'])
    return m,common.blob_read(output,m['start_bank'])


def validate_config(config, *, expected_modes=VARIANTS):
    if config['modes']!=list(expected_modes) or set(config['variants'])!=set(expected_modes):
        raise ValueError('All declared TD/repair controls required')
    for name,settings in config['variants'].items():
        if ImprovementConfig(**settings).variant!=name:raise ValueError('Variant name mismatch')
        preset=asdict(variant_config(name))
        if any(settings[k]!=preset[k] for k in ('eta_q','bootstrap','repair')):
            raise ValueError('Variant controls do not match the declared comparison')
    if set(map(int,config['seed_roles']))!=set(config['seeds']):raise ValueError('Seed roles incomplete')
    if len(set(config['seeds']))!=len(config['seeds']) or not set(config['seed_roles'].values()) <= {'development','confirmation'}:
        raise ValueError('Invalid seed roles')
    if config['checkpoints']!=[config['parent_counter']]:raise ValueError('One fixed parent per continuation required')
    if config['return_starts']<2 or config['return_horizon']<1 or config['starts']<2:
        raise ValueError('Invalid evaluation budgets')
    RevisionConfig(**config['revision'])
    if config['revision']['mode']!='normal':raise ValueError('Original growth must remain enabled in the declared comparison')


def verify_bank(env,bank):
    for start in bank['starts']:
        state=env.reset(seed=start['reset_seed'])
        for action in start['prefix_actions']:
            state,_,_,terminal,truncated=env.step(action)
            if terminal or truncated:raise ValueError('Return bank prefix ended early')
        if normalize_facts(state)!=start['state'] or physical_hash(env)!=start['physical_hash'] or ram_hash(env)!=start['ram_sha256']:
            raise ValueError('Return bank prefix/state mismatch')
        expected=env.step(1);ph=physical_hash(env)
        verify_start(env,start)
        if env.step(1)!=expected or physical_hash(env)!=ph:raise ValueError('Return bank continuation mismatch')


def extend_manifest(m,output):
    config=m['config'];evaluation=common.blob_read(output,m['start_bank']);probes=common.blob_read(output,m['validation_bank'])
    forbidden={s['ram_sha256'] for b in (evaluation,probes) for s in b['starts']}
    m['return_banks']={};m['return_diversity']={}
    env=default_factory(m['parents'][0]['cell'])()
    try:
        for point in range(config['evaluation_interval'],config['continuation_decisions']+1,config['evaluation_interval']):
            bank=make_start_bank(env,count=config['return_starts'],seed=config['return_start_seed']+point,excluded_ram=forbidden)
            verify_bank(env,bank);forbidden.update(s['ram_sha256'] for s in bank['starts'])
            m['return_banks'][str(point)]=common.blob_write(output,'return_bank_'+str(point),bank)
            m['return_diversity'][str(point)]={k:bank[k] for k in ('unique_ram','unique_encoded')}
            checked_write(output/'return_starts'/f'{point}.json',[
                {k:v for k,v in s.items() if k not in ('state','environment')} for s in bank['starts']])
    finally:env.close()
    for group in m['groups']:group['role']=config['seed_roles'][str(group['seed'])]
    m['scope']='Prospective controls and improvements; no reserved test; parents read-only'
    return m


def prepare(config_path,historical,anova,output):
    validate_config(json.loads(Path(config_path).read_text()))
    return common.prepare(config_path,historical,anova,output,expected_modes=VARIANTS,
        manifest_schema=SCHEMA,existing_loader=load_campaign,extend_manifest=extend_manifest)


def restore(folder,status,identity,env,m):
    session=common.restore_fork(folder,status,identity,env,agent_class=StableRevisionAgent,schema=FORK_SCHEMA)
    name=identity['mode'];config=m['config']
    if asdict(session.agent.improvement)!=config['variants'][name]:raise ValueError('Saved improvement settings differ')
    if asdict(session.agent.revision)!={**config['revision'],'seed':config['revision']['seed']+identity['seed']}:
        raise ValueError('Saved revision settings differ')
    return session


def return_validator(env,bank,config,point,parent_seed):
    baseline=None;attempt=0
    def validate(agent,candidate,context):
        nonlocal baseline,attempt
        before=pickle.dumps(agent,protocol=4)
        if baseline is None:
            baseline=[rollout(env,agent,s,horizon=config['return_horizon'],
                             seed=config['return_evaluation_seed']+point+s['id']) for s in bank['starts']]
        proposed=[rollout(env,candidate,s,horizon=config['return_horizon'],
                         seed=config['return_evaluation_seed']+point+s['id']) for s in bank['starts']]
        seed=[config['return_gate_seed'],parent_seed,point,attempt];attempt+=1
        gate=return_gate([v['raw_return'] for v in baseline],[v['raw_return'] for v in proposed],
                         agent.improvement,np.random.default_rng(np.random.SeedSequence(seed)))
        if pickle.dumps(agent,protocol=4)!=before:raise ValueError('Return validation mutated training agent')
        return {**gate,'baseline':baseline,'candidate':proposed,'bootstrap_seed':seed,'context':context,
                'training_updated':False,'evaluation_bank_used':False}
    return validate


def continue_one(output,m,bank,parent,name,stop):
    output=Path(output);config=m['config'];identity=common.fork_identity(m,parent,name)
    folder=output/'continuations'/parent['criterion']/str(parent['seed'])/name
    probes=common.blob_read(output,m['validation_bank'])
    with unit_lock(folder):
        env=default_factory(parent['cell'])();probe_env=default_factory(parent['cell'])()
        try:
            if (folder/'status.json').exists():
                status=checked_read(folder/'status.json');session=restore(folder,status,identity,env,m)
            else:
                session=OnlineSession.restore(common.parent_payload(parent)['session'],parent['cell'],env)
                session.agent=StableRevisionAgent.fork(session.agent,
                    RevisionConfig(**{**config['revision'],'seed':config['revision']['seed']+parent['seed']}),
                    ImprovementConfig(**config['variants'][name]))
                status=common.save_fork(folder,identity,session,None,schema=FORK_SCHEMA)
            if not common.evaluate_fork_points(output,folder,m,bank,status,probe_env,stop):return False
            goal=parent['counter']+config['continuation_decisions']
            while session.interactions<goal:
                if stop.requested:
                    if session.pending_rows:common.save_fork(folder,identity,session,status,schema=FORK_SCHEMA)
                    return False
                row=session.step();row['diagnostic']=copy.deepcopy(session.agent.last_detail)
                row['objects_after']=copy.deepcopy(env.last_auxiliary)
                point=session.interactions-parent['counter']
                if point%config['evaluation_interval']==0:
                    if session.agent.improvement.repair!='none':
                        validation=validation_probes(probe_env,probes,seed=config['probe_seed']+point,
                            blocks=config['probe_blocks'],horizon=config['probe_horizon'])
                        holdout=common.blob_write(output/'holdouts','probe_'+str(point),validation)
                        reference=common.blob_write(folder,'reference_'+str(point),{
                            'counter':session.interactions,'roots':copy.deepcopy(session.agent.root_nodes),
                            'target_roots':copy.deepcopy(session.agent.target_roots),
                            'target_counter':session.agent.target_counter,'improvement':asdict(session.agent.improvement)})
                        return_bank=common.blob_read(output,m['return_banks'][str(point)])
                        reviews=session.agent.maintain(validation,return_validator(probe_env,return_bank,config,point,parent['seed']))
                        checked_write(folder/'reviews'/f'{point}.json',{'identity':identity,'point':point,
                            'holdout':holdout,'reference':reference,'return_bank':m['return_banks'][str(point)],
                            'validation_samples':len(validation),'validation_used_for_td':False,
                            'evaluation_bank_used_for_gate':False,'reviews':reviews})
                    status=common.save_fork(folder,identity,session,status,schema=FORK_SCHEMA)
                    if not common.evaluate_fork_points(output,folder,m,bank,status,probe_env,stop):return False
                    print(json.dumps({'continued':[parent['criterion'],parent['seed'],name,point],
                                      'metrics':session.agent.revision_metrics()}),flush=True)
                elif point%config['save_interval']==0:
                    status=common.save_fork(folder,identity,session,status,schema=FORK_SCHEMA)
            checked_write(folder/'status.json',{**status,'state':'completed'})
            return True
        finally:env.close();probe_env.close()


def continue_group(output,index,stop):
    m,bank=load_campaign(output);group=m['groups'][index]
    parent=next(p for p in m['parents'] if p['criterion']==group['criterion'] and p['seed']==group['seed'])
    for name in m['config']['modes']:
        if not continue_one(output,m,bank,parent,name,stop):return False
    return True


def exact_baseline(parent,folder,status,limit):
    import gzip
    fields=('interaction','transition_id','episode_id','step_id','episode_seed','epsilon','action',
            'reward','raw_reward','terminated','truncated','episode_end','raw_episode_return','td')
    actual=[]
    for ref in status['traces']:
        actual.extend(json.loads(line) for line in gzip.decompress((folder/ref['path']).read_bytes()).splitlines())
    actual=actual[:limit];assert len(actual)==limit
    original=Path(parent['folder']);frontier=checked_read(original/'status.json');n=0
    for row in stream_rows(original,frontier['log_head']):
        if row['interaction']<=parent['counter']:continue
        if any(row[k]!=actual[n][k] for k in fields):raise ValueError('Baseline diverges from original history')
        n+=1
        if n==limit:break
    if n!=limit:raise ValueError('Original baseline history incomplete')
    return n


def report(output, *, loader=load_campaign):
    output=Path(output);m,bank=loader(output);config=m['config'];missing=[];cells=[];exact=0
    banks=[bank,common.blob_read(output,m['validation_bank'])]
    banks.extend(common.blob_read(output,ref) for ref in m['return_banks'].values())
    seen=set()
    for b in banks:
        ram={s['ram_sha256'] for s in b['starts']}
        if seen & ram:raise ValueError('Validation/evaluation banks overlap')
        seen.update(ram)
    for group in m['groups']:
        parent=next(p for p in m['parents'] if p['criterion']==group['criterion'] and p['seed']==group['seed'])
        for name in config['modes']:
            folder=output/'continuations'/group['criterion']/str(group['seed'])/name
            if not (folder/'status.json').exists():missing.append(str(folder));continue
            status=checked_read(folder/'status.json');identity=common.fork_identity(m,parent,name)
            env=default_factory(parent['cell'])()
            try:restore(folder,status,identity,env,m)
            finally:env.close()
            if status['state']!='completed' or status['continuation_decisions']!=config['continuation_decisions']:
                missing.append(str(folder));continue
            evaluations=[];events=[]
            for point in range(0,config['continuation_decisions']+1,config['evaluation_interval']):
                for start in bank['starts']:
                    path=folder/'evaluations'/str(point)/str(start['id'])/'state.json'
                    if not path.exists():missing.append(str(path));continue
                    expected={**identity,'checkpoint':status['checkpoints'][str(point)],'point':point,
                        'start_bank':m['start_bank'],'start':start['id'],'epsilon':0.,'horizon':config['evaluation_horizon'],
                        'evaluation_seed':config['evaluation_seed']+start['id']}
                    saved=common.verified_evaluation(path,expected)
                    if saved['status']!='completed':missing.append(str(path));continue
                    evaluations.append({'point':point,**saved['result']})
            for path in sorted((folder/'audits').glob('*.json'),key=lambda p:int(p.stem)):
                if path.stem not in status['checkpoints']:continue
                audit=checked_read(path)
                if audit['identity']!=identity:raise ValueError('Audit identity mismatch')
                events.extend(audit['events'])
            for kind,key in [('grow','grows_total'),('prune_review','prunes_total'),
                             ('replacement_review','replacements_total'),('refit_review','refits_total'),('target_sync','target_syncs')]:
                count=sum(e['kind']==kind and (kind in ('grow','target_sync') or e['gate']['accepted']) for e in events)
                if count!=status['metrics'][key]:raise ValueError('Improvement event counter mismatch')
            for path in (folder/'reviews').glob('*.json'):
                if path.stem not in status['checkpoints']:continue
                review=checked_read(path)
                if review['identity']!=identity or review['validation_used_for_td'] or review['evaluation_bank_used_for_gate']:
                    raise ValueError('Validation leaked into training/evaluation')
                if len(common.blob_read(output/'holdouts',review['holdout']))!=review['validation_samples']:
                    raise ValueError('Holdout size mismatch')
                ref=common.blob_read(folder,review['reference'])
                if ref['counter']!=config['parent_counter']+int(path.stem):raise ValueError('Reference frontier mismatch')
                for e in review['reviews']:
                    if e['gate']['accepted'] and name in ('guarded','stable_guarded'):
                        if not all(e[k]['accepted'] for k in ('loss_gate','policy_guard','return_guard')):
                            raise ValueError('An accepted repair bypassed a guard')
                    if 'return_guard' in e:
                        gate=e['return_guard']
                        if gate['training_updated'] or gate['evaluation_bank_used']:
                            raise ValueError('Return validator leaked into training/evaluation')
                        old,new=gate['baseline'],gate['candidate'];ib=common.blob_read(output,review['return_bank'])
                        for values in (old,new):
                            if len(values)!=len(ib['starts']):raise ValueError('Return coverage mismatch')
                            for v,s in zip(values,ib['starts']):
                                if v['physical_hash']!=s['physical_hash'] or v['seed']!=config['return_evaluation_seed']+int(path.stem)+s['id']:
                                    raise ValueError('Return start/seed mismatch')
                        reproduced=return_gate([v['raw_return'] for v in old],[v['raw_return'] for v in new],
                            ImprovementConfig(**config['variants'][name]),np.random.default_rng(np.random.SeedSequence(gate['bootstrap_seed'])))
                        if any(gate[k]!=value for k,value in reproduced.items()):raise ValueError('Return gate not reproducible')
            if name=='baseline':exact+=exact_baseline(parent,folder,status,config['continuation_decisions'])
            cells.append({**group,'mode':name,'state':status['state'],'counter':status['counter'],
                          'metrics':status['metrics'],'events':events,'evaluations':evaluations})
    result={'schema':'rrtl_improvement_report_v1','campaign':digest(m),'config':config,'complete':not missing,
            'missing':missing,'continuations':cells,'exact_baseline_transitions':exact,'banks_ram_disjoint':True,
            'scope':'Prospective improvements; no reserved test; operational gates are not general return guarantees'}
    checked_write(output/'report.json',result)
    print(json.dumps({'complete':not missing,'cells':len(cells),'missing':len(missing),'exact_baseline_transitions':exact}),flush=True)
    return not missing


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('prepare','continue','report'),required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--config',type=Path)
    parser.add_argument('--historical',type=Path);parser.add_argument('--anova',type=Path);parser.add_argument('--group',type=int,default=0)
    args=parser.parse_args()
    with StopRequest() as stop:
        if args.stage=='prepare':prepare(args.config,args.historical,args.anova,args.output);complete=True
        elif args.stage=='continue':complete=continue_group(args.output,args.group,stop)
        else:complete=report(args.output)
    raise SystemExit(0 if complete else 3)


if __name__=='__main__':main()
