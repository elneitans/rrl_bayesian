"""Read-only BART checkpoint audit. Uses only hash-verified campaign artifacts."""
from pathlib import Path
import sys, json, hashlib, pickle, csv
from collections import Counter
from dataclasses import asdict
import numpy as np

BASE = Path(__file__).resolve().parent
ROOT = next(p for p in (BASE,*BASE.parents,Path.cwd(),*Path.cwd().parents) if (p/'rrl_bayesian').is_dir())
sys.path.insert(0, str(ROOT / 'rrl_bayesian'))
from bayesian_rrtl.encoding import EncodedBatch

def checked_json(p):
    d=json.loads(p.read_text())
    if 'payload' in d:
        b=json.dumps(d['payload'], sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
        # Repository digest uses the same canonical compact JSON convention.
        from experiments.pong_analysis_runtime import digest
        assert digest(d['payload'])==d['sha256'],p
        return d['payload']
    return d

def load(p):
    b=p.read_bytes(); sha=hashlib.sha256(b).hexdigest()
    assert sha==p.stem.split('-',1)[1],p
    meta=checked_json(Path(str(p)+'.json'))
    assert meta['checkpoint']['sha256']==sha
    x=pickle.loads(b)
    assert x['counter']==meta['counter'] and x['binding']==meta['binding']
    assert x['log_head']==meta['log_head']
    return x

def rule_label(r):
    k=r.relation
    return f'{k.name}({k.obj1},{k.obj2}) {r.operator} {r.value}'

def stats(model):
    rows=[]; hist=Counter()
    for a,p in zip(model.actions,model.posteriors):
        if p is None: continue
        leaves=[]; depths=[]; stumps=0
        def visit(t,d):
            if t.is_leaf:return 1,d
            hist[(a,rule_label(t.rule))]+=1
            l,dl=visit(t.true_child,d+1);r,dr=visit(t.false_child,d+1)
            return l+r,max(dl,dr)
        for draw in p.draws:
            for t in draw.trees:
                l,d=visit(t,0);leaves.append(l);depths.append(d);stumps+=t.is_leaf
        rows.append({'action':a,'mean_leaves_per_tree':np.mean(leaves),'mean_depth_per_tree':np.mean(depths),
                     'stump_fraction':stumps/len(leaves),'max_depth':max(depths),
                     'sigma2_original_mean':np.mean(p.sigma2)*p.scale.width**2,
                     'draws':len(p.draws),'trees':p.m})
    total=sum(hist.values())
    dist={f'{a}: {r}':n/total for (a,r),n in hist.items()} if total else {}
    return rows,dist

def compare(q0,q1):
    best0=q0==q0.max(axis=1,keepdims=True);best1=q1==q1.max(axis=1,keepdims=True)
    margin0=np.sort(q0,axis=1)[:,-1]-np.sort(q0,axis=1)[:,-2]
    margin1=np.sort(q1,axis=1)[:,-1]-np.sort(q1,axis=1)[:,-2]
    return {'q_rmse':float(np.sqrt(np.mean((q1-q0)**2))),
            'q_mae':float(np.mean(abs(q1-q0))),
            'argmax_first_flip_fraction':float(np.mean(q0.argmax(1)!=q1.argmax(1))),
            'disjoint_greedy_sets_fraction':float(np.mean(~np.any(best0&best1,axis=1))),
            'q0_abs_max':float(abs(q0).max()),'q1_abs_max':float(abs(q1).max()),
            'q0_median_action_margin':float(np.median(margin0)),
            'q1_median_action_margin':float(np.median(margin1))}

def tv(h0,h1):
    if not h0 or not h1:return 0. if not h0 and not h1 else None
    return .5*sum(abs(h0.get(k,0)-h1.get(k,0)) for k in h0.keys()|h1.keys())
def dumpcsv(name,rows):
    if not rows:return
    if name=='checkpoint_comparisons.csv':
        rows=list({(r['seed'],r['kind'],r['from_model'],r['to_model']):r for r in rows}.values())
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (BASE/name).open('w',newline='') as h:
        w=csv.DictWriter(h,fieldnames=fields);w.writeheader();w.writerows(rows)

def tree_text(t,width,prefix=''):
    if t.is_leaf:return prefix+f'leaf contribution = {width*t.mu:.6g} Q units (no intercept)\n'
    return prefix+rule_label(t.rule)+'\n'+tree_text(t.true_child,width,prefix+'  True: ')+tree_text(t.false_child,width,prefix+'  False: ')

def main():
    global BASE
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=BASE/'remote',help='Extracted campaign root containing Breakout/ocatari/bart')
    parser.add_argument('--output',type=Path,default=BASE)
    args=parser.parse_args()
    source=args.source.resolve();BASE=args.output.resolve();BASE.mkdir(parents=True,exist_ok=True)
    folders=sorted((source/'Breakout/ocatari/bart').glob('*'))
    assert folders,'No extracted campaign checkpoints'
    comparisons=[];structures=[];points=[];details={};representative=[]
    for folder in folders:
        if not folder.is_dir():continue
        seed=int(folder.name)
        if not any(folder.glob('checkpoints/*.pkl')):
            print('Skipping folder without extracted checkpoints',seed,flush=True);continue
        status=checked_json(folder/'status.json');ref=status['checkpoint'];committed=set()
        while ref:
            assert ref['path'] not in committed,'Cycle in checkpoint metadata'
            committed.add(ref['path'])
            meta=checked_json(folder/(ref['path']+'.json'))
            assert meta['checkpoint']==ref and meta['binding']==status['binding']
            ref=meta['previous']
        paths=sorted([folder/p for p in committed if (folder/p).exists()],key=lambda p:int(p.name.split('-')[0]))
        loaded=[(p,load(p)) for p in paths]
        if not loaded:continue
        pool={};catalog=None;catalog_hash=None
        def addbatch(b,all_rows=True):
            nonlocal catalog,catalog_hash
            catalog,catalog_hash=b.catalog,b.catalog_hash
            for v in b.values:pool[v.tobytes()]=v.copy()
        for p,x in loaded:
            ag=x['session']['collector']['agent']
            if ag.last_fit:
                for ds in ag.last_fit.datasets:addbatch(ds.batch)
            st=x['session']['collector'].get('state')
            if isinstance(st,EncodedBatch):addbatch(st)
        if catalog is None:continue
        missing_j=[i for i,k in enumerate(catalog) if k.name=='present' and 'ball' in k.obj1 and not ('-1' in k.obj1)]
        # Include every deduplicated observed absent-ball state from retained replay.
        for p,x in loaded:
            ag=x['session']['collector']['agent']
            for tr in ag.replay.transitions:
                v=tr.state.values[0]
                if any(v[j]==0 for j in missing_j):pool[v.tobytes()]=v.copy()
        final_agent=loaded[-1][1]['session']['collector']['agent']
        reset_keys={}
        for tr in final_agent.replay.transitions:
            if tr.step_id==0:
                v=tr.state.values[0];key=v.tobytes();pool[key]=v.copy()
                reset_keys[key]=reset_keys.get(key,0)+1
        keys=sorted(pool)
        absent=[k for k in keys if any(pool[k][j]==0 for j in missing_j)]
        reset_priority=[k for k in sorted(reset_keys) if k not in set(absent)]
        priority=absent+reset_priority
        others=[k for k in keys if k not in set(priority)]
        idx=np.linspace(0,len(others)-1,min(max(0,128-len(reset_priority)),len(others)),dtype=int) if others else []
        selected=priority+[others[i] for i in idx]
        values=np.array([pool[k] for k in selected],dtype=np.int8)
        panel=EncodedBatch(values,values!=-1,catalog,catalog_hash)
        np.savez_compressed(BASE/f'panel_{seed}.npz',values=values,observed=values!=-1)
        seedinfo={'panel_size':len(panel),'absent_ball_states':len(absent),'catalog':[asdict(k) for k in catalog],
                  'checkpoints':[],'provenance':'Union last-fit X at retained checkpoints, deterministic <=128 nonabsent rows (prioritize exact step_id=0 reset states from final replay) + all absent-ball states. Unweighted support panel, not visitation-weighted.'}
        prev=None
        for p,x in loaded:
            ag=x['session']['collector']['agent']; model=ag.model; counter=x['counter']
            q=model.predict(panel); sr,hist=stats(model)
            for r in sr:structures.append({'seed':seed,'counter':counter,'model_id':model.model_id,**r})
            bound=ag.config.scale.upper
            point={'seed':seed,'counter':counter,'model_id':model.model_id,'q_abs_max':float(abs(q).max()),
                   'q_mean':float(q.mean()),'q_min':float(q.min()),'q_max':float(q.max()),
                   'fraction_abs_q_gt_100':float(np.mean(abs(q)>100)),
                   'greedy_fire_fraction':float(np.mean(q.argmax(1)==1)),
                   'absent_ball_greedy_fire_fraction':float(np.mean(q[:len(absent)].argmax(1)==1)) if absent else None}
            points.append(point);seedinfo['checkpoints'].append({**point,'rules_distribution':hist,'source':str(p.relative_to(source))})
            if prev is not None and prev[1]!=model.model_id:
                comparisons.append({'seed':seed,'kind':'retained_checkpoints','from_counter':prev[0],'to_counter':counter,
                                    'from_model':prev[1],'to_model':model.model_id,'panel_size':len(panel),
                                    'rule_distribution_TV':tv(prev[3],hist),**compare(prev[2],q)})
            prev=(counter,model.model_id,q,hist)
            old=ag.target_model
            if old.model_id!=model.model_id:
                q0=old.predict(panel);_,hist0=stats(old)
                comparisons.append({'seed':seed,'kind':'adjacent_fits_inside_checkpoint','from_counter':ag.last_fit_step-512,
                                    'to_counter':ag.last_fit_step,'from_model':old.model_id,'to_model':model.model_id,
                                    'panel_size':len(panel),'rule_distribution_TV':tv(hist0,hist),**compare(q0,q)})
            if seed==710602 and counter in (100000,500000,1000000):
                for label,mm in [('current',model)]+([('previous fit',old)] if counter==1000000 else []):
                    pp=mm.posteriors[2]
                    tt=pp.draws[-1].trees
                    def leaves(t):return 1 if t.is_leaf else leaves(t.true_child)+leaves(t.false_child)
                    jj=max(range(len(tt)),key=lambda i:leaves(tt[i]))
                    representative.append(f'\nEvolution example seed602; checkpoint {counter}; {label}; model {mm.model_id}; RIGHT; last retained draw; largest tree index{jj}. Different selected trees, NOT lineage.\n'+tree_text(tt[jj],pp.scale.width))
            if p==paths[-1]:
                # posterior uncertainty as retained draw dispersion, not convergence-certified intervals
                qdraws=np.stack([p0.predict_latent_draws(panel) for p0 in model.posteriors],axis=2)
                seedinfo['retained_draw_sd_median']=float(np.median(qdraws.std(0)))
                seedinfo['retained_draw_sd_max']=float(qdraws.std(0).max())
                seedinfo['actions']=list(model.actions)
                qo=ag.target_model.predict(panel)
                seedinfo['exact_reset_states_final_replay']=[{'values':list(map(int,v)),
                    'occurrences_in_final_replay':reset_keys[v.tobytes()],
                    'q_current':list(map(float,q[i])),'q_previous_fit':list(map(float,qo[i])),
                    'greedy_current':model.actions[int(q[i].argmax())],
                    'greedy_previous':model.actions[int(qo[i].argmax())],
                    'fire_minus_best_other_current':float(q[i,1]-max(q[i,[0,2,3]])),
                    'fire_minus_best_other_previous':float(qo[i,1]-max(qo[i,[0,2,3]])),
                    'retained_draw_sd_by_action':list(map(float,qdraws[:,i,:].std(0)))}
                    for i,v in enumerate(values) if v.tobytes() in reset_keys]
                seedinfo['absent_states_final_q']=[{'values':list(map(int,v)),'q':list(map(float,q[i])),
                    'fire_minus_best_other_q':float(q[i,1]-max(q[i,[0,2,3]])),
                    'draw_sd_by_action':list(map(float,qdraws[:,i,:].std(0))),
                    'draw_q10_by_action':list(map(float,np.quantile(qdraws[:,i,:],.1,axis=0))),
                    'draw_q90_by_action':list(map(float,np.quantile(qdraws[:,i,:],.9,axis=0)))} for i,v in enumerate(values[:len(absent)])]
                movement_j=[j for j,k in enumerate(catalog) if k.name in ('x','y') and 'ball' in k.obj1 and k.obj2 and '-1' in k.obj2]
                still=[i for i,v in enumerate(values) if movement_j and all(v[j]==1 for j in movement_j) and any(v[j]==1 for j in missing_j)]
                seedinfo['stationary_present_ball_candidate_states_final_q']=[{'values':list(map(int,values[i])),'q':list(map(float,q[i])),
                    'fire_minus_best_other_q':float(q[i,1]-max(q[i,[0,2,3]]))} for i in still]
                posterior=model.posteriors[2]
                # choose largest RIGHT tree in last draw, deterministic first tie
                trees=posterior.draws[-1].trees
                def nleaf(t):return 1 if t.is_leaf else nleaf(t.true_child)+nleaf(t.false_child)
                j=max(range(len(trees)),key=lambda i:nleaf(trees[i]))
                representative.append(f'\nSeed {seed}; model {model.model_id}; RIGHT; last retained draw; largest tree, index {j}; scale center {posterior.scale.c}, width {posterior.scale.width}\n'+tree_text(trees[j],posterior.scale.width))
            print(seed,counter,model.model_id,point['q_abs_max'],flush=True)
        details[str(seed)]=seedinfo
        dumpcsv('checkpoint_comparisons.csv',comparisons);dumpcsv('checkpoint_structures.csv',structures);dumpcsv('checkpoint_predictions.csv',points)
        (BASE/'checkpoint_summary.json').write_text(json.dumps(details,indent=2))
        (BASE/'actual_trees.txt').write_text('Representative trees are not persistent lineages.\n'+''.join(representative))
    dumpcsv('checkpoint_comparisons.csv',comparisons);dumpcsv('checkpoint_structures.csv',structures);dumpcsv('checkpoint_predictions.csv',points)
    (BASE/'checkpoint_summary.json').write_text(json.dumps(details,indent=2))
    (BASE/'actual_trees.txt').write_text('Representative trees: illustration of contributions, not standalone Q or average posterior tree. Trees chosen at different fits are NOT the same persistent tree lineage.\n'+''.join(representative))
    latest=[]
    for seed,dd in details.items():
        rr=max([r for r in comparisons if r['seed']==int(seed) and r['kind']=='adjacent_fits_inside_checkpoint'],key=lambda r:r['to_model'])
        resets=dd['exact_reset_states_final_replay']
        latest.append({'seed':seed,'model_id':rr['to_model'],'panel_size':dd['panel_size'],
            'greedy_flip_pct':100*rr['disjoint_greedy_sets_fraction'],'q_rmse':rr['q_rmse'],
            'q_mae':rr['q_mae'],'q_abs_max':rr['q1_abs_max'],'q_median_action_margin':rr['q1_median_action_margin'],
            'rule_distribution_TV':rr['rule_distribution_TV'],'reset_states':len(resets),
            'reset_greedy':';'.join(s['greedy_current'] for s in resets),
            'reset_previous_greedy':';'.join(s['greedy_previous'] for s in resets),
            'reset_fire_margin':';'.join(str(s['fire_minus_best_other_current']) for s in resets),
            'retained_draw_sd_median':dd['retained_draw_sd_median']})
    dumpcsv('checkpoint_latest_table.csv',latest)
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,1,figsize=(11,8),layout='constrained')
    for seed in details:
        rs=[r for r in comparisons if r['seed']==int(seed) and r['kind']=='adjacent_fits_inside_checkpoint']
        rs=list({r['to_model']:r for r in rs}.values())
        if rs:axes[0].plot([r['to_counter'] for r in rs],[100*r['disjoint_greedy_sets_fraction'] for r in rs],'-o',label=seed)
        ps=[r for r in points if r['seed']==int(seed)]
        ps=list({r['model_id']:r for r in ps}.values())
        axes[1].plot([r['counter'] for r in ps],[max(r['q_abs_max'],1e-6) for r in ps],'-o',label=seed)
    axes[0].set(ylabel='Estados con argmax incompatible (%)',title='Cambio entre ajustes consecutivos en tres puntos conservados')
    axes[1].set(xlabel='Decisiones de entrenamiento',ylabel='Máximo |Q| en panel fijo')
    axes[1].axhline(100,c='black',ls='--',label='Cota conservadora 100')
    for ax in axes:ax.grid(alpha=.2)
    axes[0].legend(ncol=5,fontsize=8)
    fig.savefig(BASE/'checkpoint_consistency.png',dpi=170)

if __name__=='__main__':main()
