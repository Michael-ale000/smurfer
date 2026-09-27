#!/usr/bin/env python
"""
Plot the training curves recorded by an agent's train.py.

Works for either agent -- both write the same per-round keys to
``agent_code/<agent>/training_history.pt``:

    python3 plot_training.py --agent my_agent
    python3 plot_training.py --agent dqn_agent

Four measures on very different scales (score, a rate, counts, table size) share one
x-axis, so this is drawn as small multiples rather than stacked onto twin y-axes --
a second y-scale makes two unrelated series look like they cross.  Each panel shows
the raw per-round value faintly behind a rolling mean, because the per-round signal in
Bomberman is extremely noisy and the trend is the thing worth reading.

Usage:
    python plot_training.py
    python plot_training.py --window 200 --out figures/training.png
"""

import argparse
import pickle
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent


def history_path(agent):
    return ROOT / 'agent_code' / agent / 'training_history.pt'

# Validated categorical slots; one hue per panel, identity carried by the panel title.
BLUE, ORANGE, AQUA, RED = '#2a78d6', '#eb6834', '#1baf7a', '#e34948'
INK, INK_SOFT, GRID = '#0b0b0b', '#52514e', '#e6e5e1'

PANELS = [
    ('score', 'Score per round', BLUE, '{:.2f}'),
    ('suicide', 'Suicide rate', RED, '{:.0%}'),
    ('coins', 'Coins collected', AQUA, '{:.2f}'),
    ('crates', 'Crates destroyed', ORANGE, '{:.1f}'),
]


def rolling_mean(values, window):
    if window <= 1 or len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode='valid')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', default='my_agent',
                        help='which agent_code/<agent>/training_history.pt to read')
    parser.add_argument('--history', default=None, help='override the history path')
    parser.add_argument('--window', type=int, default=0, help='0 = choose automatically')
    parser.add_argument('--out', default=None,
                        help='default: figures/<agent>_training_curves.png')
    args = parser.parse_args()

    history_file = Path(args.history) if args.history else history_path(args.agent)
    out_file = Path(args.out) if args.out else ROOT / 'figures' / f'{args.agent}_training_curves.png'

    with open(history_file, 'rb') as f:
        history = pickle.load(f)

    if not history:
        raise SystemExit('training_history.pt is empty -- run some training first.')

    rounds = np.arange(1, len(history) + 1)
    window = args.window or max(1, len(history) // 50)

    # Curriculum stages restart the stage-relative counter. The sharp steps at these
    # rounds are the scenario changing, not the agent getting worse, so mark them.
    boundaries = [i for i, h in enumerate(history)
                  if h.get('stage_round', 1) == 1 and i > 0]
    stage_labels = ['coin-heaven', 'loot-crate', 'vs weak', 'vs rule_based']

    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5), sharex=True)
    fig.patch.set_facecolor('#fcfcfb')

    for ax, (key, title, color, fmt) in zip(axes.flat, PANELS):
        raw = np.array([h.get(key, 0) for h in history], dtype=float)
        smooth = rolling_mean(raw, window)
        x_smooth = rounds[len(rounds) - len(smooth):]

        ax.plot(rounds, raw, color=color, alpha=0.16, linewidth=0.8)
        ax.plot(x_smooth, smooth, color=color, linewidth=2.0)

        # Direct end-label: several of these hues sit under 3:1 on a light surface,
        # so the number must be readable without relying on the line's colour.
        if len(smooth):
            ax.annotate(fmt.format(smooth[-1]),
                        xy=(x_smooth[-1], smooth[-1]),
                        xytext=(6, 0), textcoords='offset points',
                        va='center', ha='left', fontsize=10,
                        fontweight='bold', color=INK)

        for bx in boundaries:
            ax.axvline(bx, color=INK_SOFT, linewidth=0.9, linestyle=(0, (4, 3)),
                       alpha=0.55, zorder=1)

        ax.set_title(title, fontsize=11, color=INK, loc='left', pad=8)
        ax.set_facecolor('#fcfcfb')
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        for side in ('left', 'bottom'):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=INK_SOFT, labelsize=9, length=0)
        ax.margins(x=0.06)

    ax0 = axes.flat[0]
    starts = [0] + boundaries
    for i, sx in enumerate(starts):
        if i >= len(stage_labels):
            break
        ax0.annotate(f'{i + 1}. {stage_labels[i]}',
                     xy=(sx, 1.0), xycoords=('data', 'axes fraction'),
                     xytext=(4, -10 - 11 * (i % 2)), textcoords='offset points',
                     fontsize=8, color=INK_SOFT, ha='left', va='top')

    for ax in axes[-1]:
        ax.set_xlabel('training round', fontsize=10, color=INK_SOFT)

    fig.suptitle(f'{args.agent} training progress  ({len(history)} rounds, '
                 f'rolling mean over {window})',
                 fontsize=13, color=INK, x=0.02, ha='left', y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out = out_file
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=fig.get_facecolor())
    print(f'Wrote {out}')

    final = history[-1]
    # The tabular agent reports how many situations it has seen; the deep one reports
    # how much learning it has actually done. Print whichever the history carries.
    if 'states' in final:
        size = f"|Q| = {final['states']} situations"
    else:
        size = (f"{final.get('grad_steps', 0)} gradient steps, "
                f"buffer {final.get('buffer', 0)}")
    print(f"After {len(history)} rounds: {size}, "
          f"epsilon = {final.get('epsilon', 0):.3f}")


if __name__ == '__main__':
    main()
