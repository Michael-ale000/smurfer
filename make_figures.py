#!/usr/bin/env python
"""
Generate the report figures from measured results.

Every number here comes from a run recorded in results/ or from
agent_code/*/training_history.pt -- nothing is illustrative.

Usage:  python3 make_figures.py
"""

import pickle
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent
OUT = ROOT / 'figures'
OUT.mkdir(exist_ok=True)

# Validated categorical palette (see plot_training.py). Adjacent red/green sit in the
# 6-8 CVD band, so every series is also directly labelled -- never colour alone.
BLUE, ORANGE, AQUA, RED = '#2a78d6', '#eb6834', '#1baf7a', '#e34948'
INK, INK_SOFT, GRID, SURFACE = '#0b0b0b', '#52514e', '#e6e5e1', '#fcfcfb'

plt.rcParams.update({
    'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10,
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE,
    'axes.edgecolor': GRID, 'axes.labelcolor': INK_SOFT,
    'xtick.color': INK_SOFT, 'ytick.color': INK_SOFT,
    'text.color': INK, 'axes.grid': True, 'grid.color': GRID,
    'grid.linewidth': 0.8, 'axes.axisbelow': True,
})


def tidy(ax):
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)


# ---------------------------------------------------------------- figure 1
def fig_convergence():
    """Task 4 learning curve: score rises, suicide falls, both plateau."""
    h = pickle.load(open(ROOT / 'agent_code/dqn_agent/training_history.pt', 'rb'))
    rows = [r for i, r in enumerate(h) if 6200 <= i < 10200]        # stage 4 only
    block = 250
    n = len(rows) // block
    x = [(i * block + block // 2) for i in range(n)]
    score = [np.mean([r.get('score', 0) for r in rows[i*block:(i+1)*block]]) for i in range(n)]
    suic = [np.mean([r.get('suicide', 0) for r in rows[i*block:(i+1)*block]]) for i in range(n)]

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, y, colour, title, fmt in [
            (axes[0], score, BLUE, 'Score per round', '{:.2f}'),
            (axes[1], suic, RED, 'Rounds ending in self-destruction', '{:.0%}')]:
        ax.plot(x, y, color=colour, linewidth=2.0, marker='o', markersize=4)
        ax.annotate(fmt.format(y[-1]), xy=(x[-1], y[-1]), xytext=(6, 0),
                    textcoords='offset points', va='center', fontsize=10,
                    fontweight='bold', color=INK)
        ax.set_title(title, loc='left', color=INK, pad=8)
        ax.set_xlabel('round within task 4')
        ax.margins(x=0.12)
        tidy(ax)
    axes[1].yaxis.set_major_formatter(lambda v, _: f'{v:.0%}')
    fig.suptitle('Training on task 4 (classic, three rule_based_agents)',
                 x=0.02, ha='left', fontsize=12, color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT / 'fig_convergence.png', dpi=200)
    print('wrote figures/fig_convergence.png')


# ---------------------------------------------------------------- figure 2
def fig_comparison():
    """Greedy evaluation, 200 rounds, seed 42, three rule_based_agents."""
    agents = ['Scripted\nfeature-follower', 'DQN\n(dqn_agent)', 'Tabular Q\n(my_agent)']
    score = [3.92, 3.06, 2.50]
    kills = [0.38, 0.20, 0.23]
    suic = [0.32, 0.38, 0.28]
    colours = [ORANGE, BLUE, AQUA]

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6))
    for ax, vals, title, fmt in [
            (axes[0], score, 'Mean score per round', '{:.2f}'),
            (axes[1], kills, 'Mean kills per round', '{:.2f}'),
            (axes[2], suic, 'Suicide rate', '{:.0%}')]:
        bars = ax.bar(range(3), vals, color=colours, width=0.62)
        for b, v in zip(bars, vals):
            ax.annotate(fmt.format(v), xy=(b.get_x() + b.get_width()/2, v),
                        xytext=(0, 4), textcoords='offset points',
                        ha='center', fontsize=10, fontweight='bold', color=INK)
        ax.set_xticks(range(3)); ax.set_xticklabels(agents, fontsize=8.5)
        ax.set_title(title, loc='left', color=INK, pad=8)
        ax.margins(y=0.18); ax.grid(axis='x', visible=False)
        tidy(ax)
    axes[2].yaxis.set_major_formatter(lambda v, _: f'{v:.0%}')
    fig.suptitle('Greedy evaluation: 200 rounds, seed 42, vs three rule_based_agents',
                 x=0.02, ha='left', fontsize=12, color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(OUT / 'fig_comparison.png', dpi=200)
    print('wrote figures/fig_comparison.png')


# ---------------------------------------------------------------- figure 3
def fig_bomb_refusal():
    """The central negative result: learned agents decline safe, useful bombs."""
    labels = ['Scripted\nfeature-follower', 'DQN\n(dqn_agent)', 'Tabular Q\n(my_agent)']
    rate = [0.914, 0.582, 0.496]
    colours = [ORANGE, BLUE, AQUA]

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    bars = ax.barh(range(3), rate, color=colours, height=0.58)
    for b, v in zip(bars, rate):
        ax.annotate(f'{v:.1%}', xy=(v, b.get_y() + b.get_height()/2),
                    xytext=(6, 0), textcoords='offset points', va='center',
                    fontsize=11, fontweight='bold', color=INK)
    ax.set_yticks(range(3)); ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.05)
    ax.xaxis.set_major_formatter(lambda v, _: f'{v:.0%}')
    ax.set_xlabel('share of opportunities taken')
    ax.set_title('When the features say a bomb is safe AND would hit something,\n'
                 'how often does the agent bomb?', loc='left', color=INK, pad=10)
    ax.grid(axis='y', visible=False)
    tidy(ax)
    ax.annotate('696 such situations observed over 12 rounds',
                xy=(0, -0.24), xycoords='axes fraction', fontsize=8.5, color=INK_SOFT)
    fig.tight_layout()
    fig.savefig(OUT / 'fig_bomb_refusal.png', dpi=200)
    print('wrote figures/fig_bomb_refusal.png')


# ---------------------------------------------------------------- figure 4
def fig_compute():
    """Profiling: why the full curriculum fits in an evening on a laptop CPU."""
    labels = ['16/32/64, h256\nbatch 32\n(ours)', '16/32/64, h256\nbatch 128',
              '32/64/64, h512\nbatch 32', '32/64/64, h512\nbatch 128']
    ms = [11.2, 39.6, 17.4, 61.3]
    colours = [BLUE, GRID, GRID, ORANGE]

    fig, ax = plt.subplots(figsize=(7.6, 3.5))
    bars = ax.bar(range(4), ms, color=colours, width=0.6)
    for b, v in zip(bars, ms):
        ax.annotate(f'{v:.1f} ms', xy=(b.get_x() + b.get_width()/2, v),
                    xytext=(0, 4), textcoords='offset points', ha='center',
                    fontsize=10, fontweight='bold', color=INK)
    ax.set_xticks(range(4)); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('milliseconds')
    ax.set_title('Cost of one gradient step (CPU, 4 threads)', loc='left', color=INK, pad=8)
    ax.margins(y=0.2); ax.grid(axis='x', visible=False)
    tidy(ax)
    ax.annotate('5.5x separates our configuration from conventional defaults',
                xy=(0, -0.32), xycoords='axes fraction', fontsize=8.5, color=INK_SOFT)
    fig.tight_layout()
    fig.savefig(OUT / 'fig_compute.png', dpi=200)
    print('wrote figures/fig_compute.png')


if __name__ == '__main__':
    fig_convergence()
    fig_comparison()
    fig_bomb_refusal()
    fig_compute()
    print('\nAlso regenerate the per-stage curves:')
    print('  python3 plot_training.py --agent dqn_agent')
    print('  python3 plot_training.py --agent my_agent')
