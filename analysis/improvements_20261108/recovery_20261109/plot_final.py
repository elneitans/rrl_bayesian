"""Readable final figures; preserve the downloaded original plot unchanged."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[1]/'final'
rows = list(csv.DictReader((OUT/'evaluations.csv').open()))
assert len(rows) == 350
seeds = ['710602','710603','710607','710608','710610']
labels = {'baseline':'Baseline','slow_td':'TD lento','double_td':'Double',
          'stable_td':'TD lento + Double','guarded':'Revisión protegida',
          'stable_guarded':'TD lento + Double + revisión','refit_only':'Reestimación sin poda'}
colors = ['#111827','#2563eb','#0891b2','#059669','#d97706','#dc2626','#9333ea']
fig,axes = plt.subplots(2,5,figsize=(20,8.1))
for ri,criterion in enumerate(['historical_f','anova']):
    for ci,seed in enumerate(seeds):
        ax=axes[ri,ci]
        for (mode,label),color in zip(labels.items(),colors):
            values=sorted([r for r in rows if r['criterion']==criterion and r['seed']==seed and r['mode']==mode],key=lambda r:int(r['point']))
            assert len(values)==5
            ax.plot([int(r['point'])/1000 for r in values],[float(r['mean_return']) for r in values],
                    '-o',color=color,markersize=3,linewidth=1.7,label=label)
        ax.set(title=('Histórico' if ri==0 else 'ANOVA')+' · '+seed,
               xlabel='Miles de decisiones adicionales',ylabel='Retorno medio desde snapshot',xlim=(0,100))
        ax.set_xticks([0,25,50,75,100]);ax.set_ylim(bottom=0);ax.grid(alpha=.2)
fig.suptitle('Breakout OCAtari · 70 continuaciones completas\nCinco semillas por criterio · checkpoint 500k → 600k · doce arranques por punto',y=.995,fontsize=14)
handles,names=axes[0,0].get_legend_handles_labels()
fig.legend(handles,names,loc='upper center',bbox_to_anchor=(.5,.905),ncol=4,fontsize=10,frameon=False)
fig.subplots_adjust(left=.045,right=.995,bottom=.08,top=.78,wspace=.4,hspace=.48)
fig.savefig(OUT/'curvas_finales.png',dpi=160,bbox_inches='tight');plt.close(fig)

final=[r for r in rows if r['point']=='100000']
fig,axes=plt.subplots(2,2,figsize=(11,8.3),sharey='row')
for ri,mode in enumerate(['slow_td','refit_only']):
    for ci,criterion in enumerate(['historical_f','anova']):
        ax=axes[ri,ci]
        for i,seed in enumerate(seeds):
            values=[next(float(r['mean_return']) for r in final if r['criterion']==criterion and r['seed']==seed and r['mode']==m) for m in ['baseline',mode]]
            ax.plot([0,1],values,'-o',color=plt.cm.tab10(i),linewidth=1.8,label=seed)
        ax.set_xticks([0,1],['Baseline',labels[mode]])
        ax.set_xlim(-.15,1.15);ax.grid(axis='y',alpha=.2)
        ax.set(title=('Histórico' if ci==0 else 'ANOVA')+' · cinco semillas',ylabel='Retorno medio final desde snapshot')
    upper=max(float(r['mean_return']) for r in final if r['mode'] in ['baseline',mode])*1.08
    for ax in axes[ri]:
        ax.set_ylim(0,upper)
handles,names=axes[0,0].get_legend_handles_labels()
fig.suptitle('TD lento y reestimación: comparación pareada con baseline\nCada línea corresponde a una semilla entrenada',y=.995,fontsize=13)
fig.legend(handles,names,loc='upper center',bbox_to_anchor=(.5,.91),ncol=5,frameon=False)
fig.subplots_adjust(left=.09,right=.98,bottom=.07,top=.79,hspace=.43,wspace=.25)
fig.savefig(OUT/'comparaciones_pareadas.png',dpi=160,bbox_inches='tight');plt.close(fig)
print('Saved final curves and paired comparisons.')
