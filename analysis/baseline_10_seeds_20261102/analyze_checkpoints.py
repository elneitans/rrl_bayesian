import sys,json,pickle,hashlib,csv
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path.cwd();sys.path.insert(0,str(root/'rrl_bayesian'))
from bayesian_rrtl.encoding import CATEGORIES
from preprocessing import Fact
src=root/'tmp/baseline_analysis_20261101/remote';out=root/'rrl_bayesian/analysis/baseline_10_seeds_20261102'
def read(p):
 d=json.loads(p.read_text());h=hashlib.sha256(json.dumps(d['payload'],sort_keys=True,separators=(',',':')).encode()).hexdigest();assert h==d['sha256'];return d['payload']
def load(p):
 raw=p.read_bytes();assert hashlib.sha256(raw).hexdigest()==p.stem.split('-',1)[1];d=pickle.loads(raw);m=read(Path(str(p)+'.json'));assert d['counter']==m['counter'] and d['binding']==m['binding'];return d
rows=[];diff=[];texts=[]
for f in sorted((src/'Breakout/ocatari/corrected_common').iterdir()):
 seed=int(f.name);paths=sorted(f.glob('checkpoints/*.pkl'),key=lambda p:int(p.name.split('-')[0]));models=[load(p) for p in paths];ag=models[-1]['session']['collector']['agent']
 v=np.load(root/f'rrl_bayesian/analysis/bart_10_seeds_20261101/panel_{seed}.npz')['values'];panel=[]
 for r in v:
  panel.append(tuple(Fact(k.type,CATEGORIES[k.type][int(x)],k.name,k.obj1,k.obj2,k.obj3) for k,x in zip(ag.encoder.catalog,r) if x>=0))
 for m in models:
  s=m['session']['collector']['state'] or ();s=tuple(Fact(k.type,str(k.value) if isinstance(k.value,(bool,np.bool_)) else k.value,k.name,k.obj1,k.obj2,k.obj3) for k in s);panel.append(s)
 prev=None
 for m in models:
  a=m['session']['collector']['agent'];n=m['counter'];nodes=[];rules={};lines=[]
  def walk(t,act,path=''):
   nodes.append((act,path,t));
   if t.literal is None:lines.append(f'{act}/{path}: Q={t.q_value:.5f}');return
   label=str(tuple(t.literal));rules[(act,path)]=label;lines.append(f'{act}/{path}: {label}')
   names=['more','same','less'] if t.literal.type=='comparative' else ['true','false']
   for name in names:walk(getattr(t,name+'_branch'),act,path+'/'+name)
  for act,t in a.root_nodes.items():walk(t,act)
  q=np.array([[t.predict(s) for t in a.root_nodes.values()] for s in panel]);depth=max(t.depth for _,_,t in nodes);leaves=sum(t.literal is None for _,_,t in nodes)
  rows.append(dict(seed=seed,counter=n,splits=a.n_splits,leaves=leaves,max_depth=depth,q_min=q.min(),q_max=q.max(),root_relations=str({act:str(t.literal) for act,t in a.root_nodes.items()})))
  if prev:
   pn,pq,pr=prev;preserved=all(rules.get(k)==v for k,v in pr.items());assert preserved
   diff.append(dict(seed=seed,from_counter=pn,to_counter=n,greedy_flip_pct=100*np.mean(~np.any((q==q.max(1,keepdims=True))&(pq==pq.max(1,keepdims=True)),axis=1)),q_rmse=np.sqrt(np.mean((q-pq)**2)),old_rules_preserved=preserved,new_splits=len(rules)-len(pr)))
  prev=(n,q,rules);texts.append(f'\nSEED {seed} CHECKPOINT {n}\n'+'\n'.join(lines))
 print(seed,rows[-1],flush=True)
pd.DataFrame(rows).to_csv(out/'checkpoint_structures.csv',index=False);pd.DataFrame(diff).to_csv(out/'checkpoint_consistency.csv',index=False);(out/'trees.txt').write_text('\n'.join(texts))
fig,axs=plt.subplots(2,1,figsize=(11,7),sharex=True)
for s in sorted(set(r['seed'] for r in rows)):
 d=pd.DataFrame(rows);d=d[d.seed==s];axs[0].plot(d.counter,d.splits,'-o',label=str(s));e=pd.DataFrame(diff);e=e[e.seed==s];axs[1].plot(e.to_counter,e.greedy_flip_pct,'-o')
axs[0].set(ylabel='Divisiones acumuladas',title='RRTL: estructura persistente y cambios entre checkpoints');axs[0].legend(ncol=5,fontsize=8);axs[1].set(ylabel='Estados cuya acción cambia (%)',xlabel='Decisiones');fig.tight_layout();fig.savefig(out/'tree_evolution.png',dpi=160)
