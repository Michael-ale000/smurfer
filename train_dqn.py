#!/usr/bin/env python
"""
Curriculum trainer for agent_code/dqn_agent (deep Q-network).

Same argument as for the tabular agent, only sharper: a randomly initialised network
that starts on the full four-opponent board dies to its own first bomb every round and
learns one lesson very thoroughly and nothing else.  So we walk it through the four
tasks from the project sheet, each a superset of the last, carrying the *network* --
and Adam's moment estimates -- forward between stages.

    stage 1  coin-heaven, alone          navigate and collect revealed coins
    stage 2  loot-crate, alone           use bombs on crates without dying
    stage 3  classic, weak opponents     survive a shared board, hunt easy targets
    stage 4  classic, rule_based_agent   full-strength opposition

Exploration is annealed inside each stage and restarted lower in every later stage, so
a new stage perturbs the policy without wiping out what the previous one taught.

The replay buffer does *not* survive a stage boundary (each stage is a separate
process), which is deliberate: stage 3's buffer full of solo-play transitions would be
off-distribution for stage 4 anyway.  The first few hundred rounds of each stage refill
it, which is why LEARN_START is small.

Usage:
    python3 train_dqn.py                    # full curriculum from scratch
    python3 train_dqn.py --stages 3 4       # later stages only, keeping the network
    python3 train_dqn.py --scale 0.25       # a quarter of the rounds everywhere
    python3 train_dqn.py --arch mlp         # the no-convolution ablation
    python3 train_dqn.py --quick            # 40 rounds per stage, pipeline check only
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

AGENT = 'dqn_agent'
AGENT_DIR = Path(__file__).parent / 'agent_code' / AGENT
MODEL = AGENT_DIR / 'dqn_model.pt'
HISTORY = AGENT_DIR / 'training_history.pt'

# rounds, scenario, opponents, exploration schedule
STAGES = {
    1: dict(rounds=1200, scenario='coin-heaven', opponents=[],
            eps_start=0.90, eps_end=0.10),
    2: dict(rounds=2500, scenario='loot-crate', opponents=[],
            eps_start=0.50, eps_end=0.08),
    3: dict(rounds=2500, scenario='classic',
            opponents=['peaceful_agent', 'coin_collector_agent', 'coin_collector_agent'],
            eps_start=0.35, eps_end=0.06),
    4: dict(rounds=4000, scenario='classic',
            opponents=['rule_based_agent', 'rule_based_agent', 'rule_based_agent'],
            eps_start=0.25, eps_end=0.04),
}


def run_stage(number, cfg, arch, rounds, extra_env):
    env = dict(os.environ)
    env['DQN_QUIET'] = '1'               # per-step DEBUG logging dominates otherwise
    env['DQN_ARCH'] = arch
    env['DQN_EPS_START'] = str(cfg['eps_start'])
    env['DQN_EPS_END'] = str(cfg['eps_end'])
    # Anneal across ~80% of the stage so it finishes at the floor and consolidates.
    env['DQN_EPS_DECAY'] = str(max(1, int(rounds * 0.8)))
    env['DQN_SAVE_EVERY'] = str(min(100, max(1, rounds // 4)))
    env.update(extra_env)

    cmd = [sys.executable, 'main.py', 'play', '--no-gui',
           '--agents', AGENT, *cfg['opponents'],
           '--train', '1',
           '--scenario', cfg['scenario'],
           '--n-rounds', str(rounds)]

    label = cfg['scenario'] + (f" vs {', '.join(cfg['opponents'])}" if cfg['opponents'] else ' (solo)')
    print(f"\n{'=' * 70}\nSTAGE {number}: {label}\n"
          f"  {rounds} rounds, epsilon {cfg['eps_start']} -> {cfg['eps_end']}, arch={arch}\n"
          f"{'=' * 70}", flush=True)

    t0 = time.time()
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        print(f"Stage {number} FAILED (exit {result.returncode})", file=sys.stderr)
        sys.exit(result.returncode)
    print(f"Stage {number} finished in {time.time() - t0:.0f}s "
          f"({(time.time() - t0) / max(1, rounds):.2f}s per round)", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--stages', type=int, nargs='+', default=sorted(STAGES),
                        choices=sorted(STAGES), help='which curriculum stages to run')
    parser.add_argument('--arch', default='conv', choices=['conv', 'mlp'],
                        help='network torso (conv is the real agent, mlp the ablation)')
    parser.add_argument('--scale', type=float, default=1.0,
                        help='multiply every stage length, e.g. 0.25 for a short run')
    parser.add_argument('--fresh', action='store_true',
                        help='move the existing model aside and start over')
    parser.add_argument('--quick', action='store_true',
                        help='40 rounds per stage, just to check the pipeline runs')
    parser.add_argument('--device', default=None,
                        help='cpu (default), mps or cuda; the net is small enough that '
                             'CPU is usually fastest')
    parser.add_argument('--threads', default=None, help='torch CPU threads')
    args = parser.parse_args()

    if args.fresh:
        for path in (MODEL, HISTORY):
            if path.exists():
                backup = path.with_suffix('.pt.bak')
                shutil.move(str(path), str(backup))
                print(f'Moved existing {path.name} to {backup.name}')

    extra_env = {}
    if args.device:
        extra_env['DQN_DEVICE'] = args.device
    if args.threads:
        extra_env['DQN_THREADS'] = args.threads

    for number in args.stages:
        cfg = STAGES[number]
        rounds = 40 if args.quick else max(1, int(cfg['rounds'] * args.scale))
        run_stage(number, cfg, args.arch, rounds, extra_env)

    if MODEL.exists():
        print(f"\nDone. Model at {MODEL} ({MODEL.stat().st_size / 1e6:.1f} MB)")
        print("Evaluate it with:  python3 evaluate_agent.py --agent dqn_agent")
    else:
        print("\nWarning: no model was written.", file=sys.stderr)


if __name__ == '__main__':
    main()
