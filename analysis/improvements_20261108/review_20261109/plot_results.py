"""Scientific figures from the checksummed partial campaign review."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
rows = list(csv.DictReader((ROOT/'curves.csv').open()))
labels = {'baseline': 'Baseline', 'slow_td': 'TD lento', 'double_td': 'Double',
          'stable_td': 'TD lento + Double', 'guarded': 'Revisión protegida',
          'stable_guarded': 'TD lento + Double + revisión', 'refit_only': 'Reestimación sin poda'}
colors = ['#111827', '#2563eb', '#0891b2', '#059669', '#d97706', '#dc2626', '#9333ea']
seeds = ['710602', '710603', '710607', '710608', '710610']
fig, axes = plt.subplots(2, 5, figsize=(20, 7.8))
for ri, criterion in enumerate(['historical_f', 'anova']):
    for ci, seed in enumerate(seeds):
        ax = axes[ri, ci]
        for (mode, label), color in zip(labels.items(), colors):
            rs = sorted([v for v in rows if v['criterion']==criterion and v['seed']==seed and v['mode']==mode], key=lambda r:int(r['point']))
            if not rs:
                continue
            ax.plot([int(v['point'])/1000 for v in rs], [float(v['mean_return']) for v in rs],
                    '-o', color=color, label=label, markersize=3, linewidth=1.6)
        title = ('Histórico' if ri == 0 else 'ANOVA')+' · '+seed
        if ri == 1 and seed in ['710608','710610']:
            title += ' (incompleto)'
        ax.set(title=title, xlabel='Miles de decisiones nuevas', ylabel='Retorno medio desde snapshot', xlim=(0,100))
        ax.set_xticks([0,25,50,75,100])
        ax.set_ylim(bottom=0)
        ax.grid(alpha=.2)
handles, names = axes[0,0].get_legend_handles_labels()
fig.suptitle('Breakout OCAtari · continuaciones desde 500k\nCampaña parcial: 61/70 continuaciones completas; 12 arranques por punto', y=.995, fontsize=14)
fig.legend(handles, names, loc='upper center', bbox_to_anchor=(.5,.925), ncol=4, frameon=False, fontsize=10)
fig.tight_layout(rect=(0,0,1,.845))
fig.savefig(ROOT/'curvas_parciales.png', dpi=160, bbox_inches='tight')
plt.close(fig)

final = list(csv.DictReader((ROOT/'final.csv').open()))
fig, axes = plt.subplots(1,2,figsize=(11,4.6),sharey=True)
for ax, criterion in zip(axes,['historical_f','anova']):
    for i, seed in enumerate(seeds):
        a = next(float(v['mean_return']) for v in final if v['criterion']==criterion and v['seed']==seed and v['mode']=='baseline')
        b = next(float(v['mean_return']) for v in final if v['criterion']==criterion and v['seed']==seed and v['mode']=='slow_td')
        ax.plot([0,1],[a,b],'-o', color=plt.cm.tab10(i), label=seed, linewidth=1.8)
    ax.set_xticks([0,1],['Baseline\nη = .025','TD lento\nη = .005'])
    ax.set_xlim(-.15,1.15)
    ax.set_title('Histórico · 5 semillas' if criterion=='historical_f' else 'ANOVA · 5 semillas')
    ax.set_ylabel('Retorno medio final desde snapshot')
    ax.grid(axis='y',alpha=.2)
handles,names=axes[0].get_legend_handles_labels()
fig.legend(handles,names,loc='upper center',bbox_to_anchor=(.5,.94),ncol=5,frameon=False)
fig.suptitle('Comparación completa disponible: bajar únicamente el paso TD',y=1.01)
fig.tight_layout(rect=(0,0,1,.86))
fig.savefig(ROOT/'td_lento_pareado.png',dpi=160,bbox_inches='tight')
plt.close(fig)
print('Saved curves and matched TD comparison.')
