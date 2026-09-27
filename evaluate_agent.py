#!/usr/bin/env python
"""
Benchmark a trained agent against the provided agents, with learning switched off.

Runs a match through main.py with --save-stats, then reports the numbers that actually
matter for the tournament: mean score, how often we come first, and how often we kill
ourselves.  Suicide rate is worth watching separately -- an agent can have a decent
average score while throwing away a third of its rounds to its own bombs.

Usage:
    python evaluate_agent.py                          # 200 rounds vs 3 rule_based_agents
    python evaluate_agent.py --opponents coin_collector_agent --n-rounds 100
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent


def run_match(agent, opponents, n_rounds, scenario, seed=None):
    match_name = f'eval-{datetime.now().strftime("%Y%m%d-%H%M%S-%f")}'
    cmd = [sys.executable, 'main.py', 'play', '--no-gui',
           '--agents', agent, *opponents,
           '--n-rounds', str(n_rounds),
           '--scenario', scenario,
           '--save-stats', f'results/{match_name}.json',
           '--match-name', match_name]
    if seed is not None:
        cmd += ['--seed', str(seed)]

    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        sys.exit(result.returncode)

    with open(ROOT / 'results' / f'{match_name}.json') as f:
        return json.load(f)


def report(stats, agent, n_rounds):
    by_agent = stats['by_agent']

    # The framework suffixes duplicate agent names, so match on the prefix.
    key = next(k for k in by_agent if k == agent or k.startswith(agent + '_'))
    me = by_agent[key]

    print(f"\n{'=' * 62}")
    print(f"{'agent':<28}{'score':>8}{'coins':>8}{'kills':>8}{'suicides':>10}")
    print('-' * 62)
    for name, s in sorted(by_agent.items(), key=lambda kv: -kv[1].get('score', 0)):
        marker = ' *' if name == key else '  '
        print(f"{marker}{name:<26}{s.get('score', 0):>8}{s.get('coins', 0):>8}"
              f"{s.get('kills', 0):>8}{s.get('suicides', 0):>10}")
    print('-' * 62)

    rounds = max(1, n_rounds)
    print(f"\nPer round over {n_rounds} rounds:")
    print(f"  mean score      {me.get('score', 0) / rounds:6.2f}")
    print(f"  mean coins      {me.get('coins', 0) / rounds:6.2f}")
    print(f"  mean kills      {me.get('kills', 0) / rounds:6.2f}")
    print(f"  suicide rate    {me.get('suicides', 0) / rounds:6.1%}")

    best_other = max((s.get('score', 0) for n, s in by_agent.items() if n != key), default=0)
    print(f"\n  our total {me.get('score', 0)} vs best opponent total {best_other}"
          f"  ->  {'AHEAD' if me.get('score', 0) > best_other else 'BEHIND'}")
    print('=' * 62)
    return me


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--agent', default='my_agent')
    parser.add_argument('--opponents', nargs='*', default=['rule_based_agent'] * 3)
    parser.add_argument('--n-rounds', type=int, default=200)
    parser.add_argument('--scenario', default='classic')
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()

    (ROOT / 'results').mkdir(exist_ok=True)
    stats = run_match(args.agent, args.opponents, args.n_rounds, args.scenario, args.seed)
    report(stats, args.agent, args.n_rounds)


if __name__ == '__main__':
    main()
