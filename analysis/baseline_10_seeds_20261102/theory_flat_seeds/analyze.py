"""Static, hash-verified tree audit. Counterfactual probes, no emulator or training."""
from pathlib import Path
import sys,json,pickle,hashlib
import pandas as pd
import numpy as np
from scipy.stats import f
ROOT=Path.cwd();sys.path.insert(0,str(ROOT/'rrl_bayesian'))
from preprocessing import Fact
SRC=ROOT/'tmp/baseline_analysis_20261101/remote/Breakout/ocatari/corrected_common'
OUT=Path(__file__).resolve().parent

def read(p):
 d=json.loads(p.read_text());assert hashlib.sha256(json.dumps(d['payload'],sort_keys=True,separators=(',',':')).encode()).hexdigest()==d['sha256'];return d['payload']
def load(p):
 raw=p.read_bytes();assert hashlib.sha256(raw).hexdigest()==p.stem.split('-',1)[1];m=read(Path(str(p)+'.json'));d=pickle.loads(raw);assert m['binding']==d['binding'] and m['counter']==d['counter'];return d
rows=[];stats=[];nodes=[]
for seed in range(710601,710611):
 folder=SRC/str(seed);status=read(folder/'status.json');payload=load(folder/status['checkpoint']['path']);ag=payload['session']['collector']['agent']
 def walk(node,act,path):
  nodes.append(dict(seed=seed,action=act,path=path,depth=node.depth,split=str(node.literal),q=node.q_value,terminal=node.literal is None))
  if node.literal is not None:
   for name in ['more','same','less'] if node.literal.type=='comparative' else ['true','false']:walk(getattr(node,name+'_branch'),act,path+'/'+name)
  else:
   for lit,groups in node.refinements_stats.items():
    n=groups['Total']['n'];J=groups['Total']['J'];W=sum(d['J'] for k,d in groups.items() if k!='Total');ratio=W/J if J>0 else None;p=float(f.cdf(ratio,n-1,n-1)) if J>0 and n>1 else None
    stats.append(dict(seed=seed,action=act,path=path,depth=node.depth,literal=str(lit),n=n,J=J,W=W,ratio=ratio,p=p,eligible=n>node.min_sample_size and J>0,groups=groups))
 for act,t in ag.root_nodes.items():walk(t,act,'root')
 for pos in ['less','same','more']:
  for move in ['less','same','more']:
   state=[]
   for k in ag.encoder.catalog:
    value='True' if k.name=='present' else 'False' if k.name=='incontact' else 'same'
    if k.name=='x' and k.obj1=='player_t' and k.obj2=='ball_t':value=pos
    if k.name=='x' and k.obj1=='player_t' and k.obj2=='player_t-1':value=move
    if k.name=='y' and k.obj1=='player_t' and k.obj2=='ball_t':value='more'
    if k.obj1=='ball_t' and k.obj2=='ball_t-1':value='more'
    state.append(Fact(k.type,value,k.name,k.obj1,k.obj2,k.obj3))
   q=ag.predict_q(state);best=[a for a,v in zip(ag.actions,q) if v==max(q)]
   rows.append(dict(seed=seed,position=pos,previous_player_motion=move,greedy='/'.join(best),**dict(zip(ag.actions,map(float,q)))))
pd.DataFrame(rows).to_csv(OUT/'policy_probe.csv',index=False)
pd.DataFrame(nodes).to_csv(OUT/'nodes.csv',index=False)
(OUT/'leaf_statistics.json').write_text(json.dumps(stats,indent=2))
for seed in [710602,710608,710610,710607]:
 print('\nSEED',seed)
 print(pd.DataFrame(rows).query('seed==@seed').to_string(index=False))
 for r in stats:
  if r['seed']==seed and r['action']=='RIGHT' and "obj2='ball_t'" in r['literal'] and "name='x'" in r['literal']:
   print('candidate x(player,ball)',{k:v for k,v in r.items() if k!='groups'},'groups',r['groups'])
