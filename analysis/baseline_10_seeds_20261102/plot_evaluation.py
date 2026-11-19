"""Read-only evaluation audit and plots; snapshots, never interpolated learning curves."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def read(path):
    doc = json.loads(path.read_text())
    raw = json.dumps(doc['payload'], sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    assert hashlib.sha256(raw).hexdigest() == doc['sha256'], path
    return doc['payload']


def main(root, out):
    out.mkdir(parents=True, exist_ok=True)
    manifest = read(root/'campaign.json')
    campaign_hash=hashlib.sha256(json.dumps(manifest,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    rows, episodes = [], []
    for cell in manifest['cells']:
        folder = root/cell['id']
        status = read(folder/'status.json')
        assert status['binding']['cell'] == cell['hash']
        assert status['binding']['campaign'] == campaign_hash
        seed = cell['seed']
        results = []
        for p in sorted((folder/'evaluations').glob('*/*/*/state.json')):
            state, identity = read(p), read(p.with_name('identity.json'))
            assert state['identity_sha256'] == hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            assert identity['campaign']==campaign_hash and identity['cell']==cell['hash']
            assert identity['group']==p.parts[-4] and identity['seed']==int(p.parts[-2])
            ref=identity['checkpoint']
            assert ref['path']==f'checkpoints/{p.parts[-3]}-{ref["sha256"]}.pkl'
            if int(p.parts[-3])==status['counter']:
                assert ref==status['checkpoint']
            if state['status'] != 'completed':
                continue
            r = state['result']
            e = dict(seed=seed, group=p.parts[-4], interaction=int(p.parts[-3]), eval_seed=int(p.parts[-2]), **{k:v for k,v in r.items() if k!='seed'})
            episodes.append(e)
            if e['group']=='final_development' and e['interaction']==1000000:
                results.append(r)
        metrics=status['metrics']
        rows.append(dict(seed=seed, interaction=status['counter'], completed_training_games=metrics['completed_games'],
                         evaluations=len(results), mean_return=np.mean([r['raw_return'] for r in results]) if results else np.nan,
                         truncations=sum(r['truncated'] for r in results),
                         **{a:sum(r['actions'].get(a,0) for r in results) for a in cell['learning']['actions']}))
    df=pd.DataFrame(rows).sort_values('seed')
    pd.DataFrame(episodes).to_json(out/'evaluation_episodes.json',orient='records',indent=2)
    df.to_csv(out/'evaluation_summary.csv',index=False)
    fig,axs=plt.subplots(1,2,figsize=(13,4.7),gridspec_kw={'width_ratios':[1.2,1]})
    colors=['#2563eb' if n==10 else '#f59e0b' for n in df.evaluations]
    vals=df.mean_return.fillna(0)
    axs[0].bar(np.arange(10),vals,color=colors)
    for i,(_,r) in enumerate(df.iterrows()):
        label=f'{r.mean_return:g}' if r.evaluations else 'sin eval.'
        if r.evaluations and r.evaluations<10: label+=f'\n({int(r.evaluations)}/10)'
        axs[0].text(i,max(vals.iloc[i],0)+.6,label,ha='center',fontsize=9)
    axs[0].set(xticks=range(10),xticklabels=[str(x)[-3:] for x in df.seed],xlabel='Semilla (prefijo 710)',ylabel='Retorno bruto medio',title='Evaluación greedy a 1M de decisiones',ylim=(-1,115))
    actions=['NOOP','FIRE','RIGHT','LEFT']
    counts=df[actions].to_numpy(float); denom=counts.sum(axis=1,keepdims=True)
    frac=np.divide(counts,denom,out=np.full_like(counts,np.nan),where=denom>0)
    im=axs[1].imshow(frac,vmin=0,vmax=1,cmap='Blues',aspect='auto')
    for i in range(10):
        for j in range(4):
            if np.isfinite(frac[i,j]): axs[1].text(j,i,f'{frac[i,j]*100:.1f}%',ha='center',va='center',color='white' if frac[i,j]>.6 else 'black',fontsize=8)
    axs[1].set(xticks=range(4),xticklabels=actions,yticks=range(10),yticklabels=[str(x) for x in df.seed],title='Fracción de acciones en evaluación')
    fig.colorbar(im,ax=axs[1],label='Fracción')
    fig.suptitle('RRTL baseline · Breakout OCAtari · instantánea de resultados disponibles',fontsize=13)
    fig.tight_layout(); fig.savefig(out/'evaluation_final.png',dpi=170);plt.close(fig)
    ep=pd.DataFrame(episodes)
    agg=ep.groupby(['seed','group','interaction']).agg(mean_return=('raw_return','mean'),n=('raw_return','size'),truncated=('truncated','sum')).reset_index()
    agg.to_csv(out/'evaluation_history.csv',index=False)
    fig,axes=plt.subplots(2,5,figsize=(16,7),sharex=True,sharey=True)
    for ax,seed in zip(axes.flat,df.seed):
        sub=agg[agg.seed==seed]
        inter=sub[sub.group=='intermediate'].sort_values('interaction')
        # Join only contiguous scheduled intermediate checkpoints; missing units remain gaps.
        grid=np.array([0,100000,500000,1000000])
        lookup=dict(zip(inter.interaction,inter.mean_return))
        ax.plot(grid,[lookup.get(int(x),np.nan) for x in grid],'-o',color='#2563eb',label='Intermedia (2 partidas)')
        fin=sub[sub.group=='final_development']
        for _,r in fin.iterrows():
            ax.scatter(r.interaction,r.mean_return,marker='x',s=75,color='#dc2626',zorder=5)
            ax.annotate(f'{r.mean_return:g} ({int(r.n)}/10)',(r.interaction,r.mean_return),xytext=(-5,10),textcoords='offset points',ha='right',fontsize=8)
        ax.set(title=str(seed),ylim=(-2,115),xticks=[0,500000,1000000],xticklabels=['0','500k','1M'])
        ax.grid(alpha=.2)
    for ax in axes[:,0]:ax.set_ylabel('Retorno bruto medio greedy')
    for ax in axes[-1,:]:ax.set_xlabel('Decisiones de entrenamiento')
    fig.suptitle('Evaluación sin exploración: azul = calendario intermedio; × roja = final de desarrollo\nLos huecos son evaluaciones no completadas; las líneas solo guían la vista.',fontsize=12)
    fig.tight_layout();fig.savefig(out/'evaluation_evolution.png',dpi=170);plt.close(fig)
    print(df.to_string(index=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('out',type=Path)
    a=p.parse_args();main(a.root,a.out)
