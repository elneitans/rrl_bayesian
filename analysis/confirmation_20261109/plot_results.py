"""Readable figures for the completed prospective ANOVA confirmation."""
import argparse
import csv
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MODES = ['baseline', 'slow_td', 'refit_only']
LABELS = {'baseline': 'Baseline', 'slow_td': 'TD lento', 'refit_only': 'Reestimación sin poda'}
COLORS = {'baseline': '#252f3f', 'slow_td': '#3164c9', 'refit_only': '#008d96'}
SEEDS = list(range(721101, 721106))


def plot(analysis, output):
    rows = list(csv.DictReader((analysis / 'evaluations.csv').open()))
    assert len(rows) == 315
    curves = {(s, m): sorted([r for r in rows if int(r['seed']) == s and r['mode'] == m],
                             key=lambda r: int(r['point'])) for s in SEEDS for m in MODES}
    assert all(len(c) == 21 for c in curves.values())
    output.mkdir(parents=True, exist_ok=True)
    upper = max(float(r['mean_return']) for r in rows) * 1.06
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.grid': True, 'grid.alpha': .2})
    fig, axes = plt.subplots(3, 2, figsize=(11, 10), sharex=True, sharey=True)
    for axis, seed in zip(axes.flat, SEEDS):
        for mode in MODES:
            curve = curves[seed, mode]
            axis.plot([(.5 + int(r['point']) / 1e6) for r in curve],
                      [float(r['mean_return']) for r in curve], color=COLORS[mode],
                      marker='o', markersize=3, linewidth=1.5, label=LABELS[mode])
        axis.set_title(f'Semilla {seed}')
        axis.set_xlim(.5, 1)
        axis.set_ylim(-3, max(1, upper))
        axis.set_xlabel('Millones de decisiones acumuladas')
        axis.set_ylabel('Retorno medio desde snapshot')
        axis.tick_params(labelbottom=True)
    axes.flat[-1].axis('off')
    axes.flat[-1].text(.05, .7,
        'Cada punto: media de 12 arranques compartidos.\n'
        'Horizonte: 3.000 decisiones, política greedy.\n'
        'Cinco semillas nuevas; mismo padre por variante.\n\n'
        'Las líneas unen observaciones cada 25k.\n'
        'Los 12 arranques no son 12 entrenamientos.\n'
        'No son puntajes de una partida desde reset.', transform=axes.flat[-1].transAxes,
        fontsize=11, va='top', linespacing=1.7)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .945), ncol=3, frameon=False)
    fig.suptitle('Breakout / OCAtari: seguimiento de 500k a 1M', y=.98, fontsize=16)
    fig.subplots_adjust(top=.89, bottom=.07, hspace=.4, wspace=.22)
    fig.savefig(output / 'curvas_finales.png', dpi=180)
    plt.close(fig)

    final = {(s, m): float(curves[s, m][-1]['mean_return']) for s in SEEDS for m in MODES}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    palette = ['#9a4f00', '#56803b', '#7355a0', '#b13250', '#587780']
    for seed, color in zip(SEEDS, palette):
        axes[0].plot(range(3), [final[seed, m] for m in MODES], 'o-', alpha=.8,
                     color=color, linewidth=1, label=str(seed))
    means = [statistics.mean(final[s, m] for s in SEEDS) for m in MODES]
    axes[0].scatter(range(3), means, marker='D', color='black', s=48, zorder=5, label='Media')
    axes[0].set_xticks(range(3), ['Baseline', 'TD lento', 'Reestimación'])
    axes[0].set_ylabel('Retorno final medio de los 12 arranques')
    axes[0].set_title('Endpoint fijado: 1M')
    axes[0].legend(fontsize=8, ncol=2, frameon=False)
    for j, mode in enumerate(MODES[1:]):
        differences = [final[s, mode] - final[s, 'baseline'] for s in SEEDS]
        for offset, value, color in zip([-.12, -.06, 0, .06, .12], differences, palette):
            axes[1].scatter(j + offset, value, color=color, s=45, zorder=3)
        axes[1].scatter(j, statistics.mean(differences), marker='D', color='black', s=48, zorder=5)
    axes[1].axhline(0, color='#555', linewidth=1)
    axes[1].set_xlim(-.5, 1.5)
    axes[1].set_xticks([0, 1], ['TD lento', 'Reestimación'])
    axes[1].set_ylabel('Diferencia pareada respecto del baseline')
    axes[1].set_title('Una observación por semilla entrenada')
    fig.suptitle('Confirmación prospectiva: resultados finales y heterogeneidad', fontsize=14)
    fig.text(.5, .025, 'Colores: cinco semillas. Diamante negro: media. Sin selección retrospectiva de checkpoints.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .06, 1, .93))
    fig.savefig(output / 'comparaciones_finales.png', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    plot(args.analysis, args.output)
