#!/usr/bin/env python
"""
Curriculum trainer for agent_code/my_agent.

Training the full four-opponent game from scratch does not work well: the agent dies
to its own first bomb long before it ever sees a coin, so almost every episode ends in
the same uninformative way.  Instead we walk it through the four tasks from the project
sheet, each one a superset of the last, carrying the Q-table forward between stages.

    stage 1  coin-heaven, alone          navigate and collect revealed coins
    stage 2  loot-crate, alone           use bombs on crates without dying
    stage 3  classic, weak opponents     survive a shared board, hunt easy targets
    stage 4  classic, rule_based_agent   full-strength opposition

Exploration is annealed inside each stage and restarted lower in every later stage, so
a new stage perturbs the policy without wiping out what the previous one taught.

Usage:
    python train_agent.py                 # full curriculum from scratch
    python train_agent.py --stages 3 4    # only the later stages, keeping the Q-table
    python train_agent.py --quick         # tiny run to check the pipeline works
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

AGENT = 'my_agent'
AGENT_DIR = Path(__file__).parent / 'agent_code' / AGENT
Q_TABLE = AGENT_DIR / 'q_table.pt'
HISTORY = AGENT_DIR / 'training_history.pt'

# rounds, scenario, opponents, eps_start, eps_end, eps_decay fraction of the stage
STAGES = {
    1: dict(rounds=3000, scenario='coin-heaven', opponents=[],
            eps_start=0.85, eps_end=0.10),
    2: dict(rounds=6000, scenario='loot-crate', opponents=[],
            eps_start=0.50, eps_end=0.08),
    3: dict(rounds=6000, scenario='classic',
            opponents=['peaceful_agent', 'coin_collector_agent', 'coin_collector_agent'],
            eps_start=0.35, eps_end=0.06),
    4: dict(rounds=10000, scenario='classic',
            opponents=['rule_based_agent', 'rule_based_agent', 'rule_based_agent'],
            eps_start=0.25, eps_end=0.04),
}


def run_stage(number, cfg, quick=False):
    rounds = 40 if quick else cfg['rounds']

    env = dict(os.environ)
    env['MY_AGENT_QUIET'] = '1'          # per-step DEBUG logging dominates otherwise
    env['MY_AGENT_EPS_START'] = str(cfg['eps_start'])
    env['MY_AGENT_EPS_END'] = str(cfg['eps_end'])
    # Anneal across ~80% of the stage so it finishes at the floor and consolidates.
    env['MY_AGENT_EPS_DECAY'] = str(max(1, int(rounds * 0.8)))
    env['MY_AGENT_SAVE_EVERY'] = str(min(250, max(1, rounds // 4)))

    cmd = [sys.executable, 'main.py', 'play', '--no-gui',
           '--agents', AGENT, *cfg['opponents'],
           '--train', '1',
           '--scenario', cfg['scenario'],
           '--n-rounds', str(rounds)]

    label = cfg['scenario'] + (f" vs {', '.join(cfg['opponents'])}" if cfg['opponents'] else ' (solo)')
    print(f"\n{'=' * 70}\nSTAGE {number}: {label}\n"
          f"  {rounds} rounds, epsilon {cfg['eps_start']} -> {cfg['eps_end']}\n{'=' * 70}",
          flush=True)

    t0 = time.time()
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        print(f"Stage {number} FAILED (exit {result.returncode})", file=sys.stderr)
        sys.exit(result.returncode)
    print(f"Stage {number} finished in {time.time() - t0:.0f}s", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--stages', type=int, nargs='+', default=sorted(STAGES),
                        choices=sorted(STAGES), help='which curriculum stages to run')
    parser.add_argument('--fresh', action='store_true',
                        help='delete the existing Q-table and start over')
    parser.add_argument('--quick', action='store_true',
                        help='40 rounds per stage, just to check the pipeline runs')
    args = parser.parse_args()

    if args.fresh:
        for path in (Q_TABLE, HISTORY):
            if path.exists():
                backup = path.with_suffix('.pt.bak')
                shutil.move(str(path), str(backup))
                print(f'Moved existing {path.name} to {backup.name}')

    for number in args.stages:
        run_stage(number, STAGES[number], quick=args.quick)

    if Q_TABLE.exists():
        size_mb = Q_TABLE.stat().st_size / 1e6
        print(f"\nDone. Q-table at {Q_TABLE} ({size_mb:.1f} MB)")
    else:
        print("\nWarning: no Q-table was written.", file=sys.stderr)


if __name__ == '__main__':
    main()
