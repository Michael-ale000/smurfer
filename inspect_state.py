#!/usr/bin/env python
"""
Print one game state as the agent actually receives it.

Study aid, not part of any agent. `act(self, game_state)` is handed a plain dict once
per step, and every feature in both agents is a pure function of that dict -- so the
fastest way into this codebase is to look at one.

Usage:
    python3 inspect_state.py                # state at step 20 of a rule_based game
    python3 inspect_state.py --step 60      # later, once bombs are flying
    python3 inspect_state.py --agent dqn_agent --features
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import settings as s
from environment import BombeRLeWorld
from argparse import Namespace


def make_world(agent, opponents):
    args = Namespace(
        command_name='play', save_replay=False, no_gui=True, save_stats=False,
        match_name='inspect', seed=42, silence_errors=False, log_dir='logs',
        continue_without_training=True, scenario='classic', make_video=False,
        n_rounds=1, update_interval=0.1, turn_based=False, train=0,
    )
    agents = [(agent, False)] + [(o, False) for o in opponents]
    return BombeRLeWorld(args, agents)


def draw(state):
    """The board as ASCII, from our point of view."""
    field = state['field']
    bombs = {pos: t for pos, t in state['bombs']}
    coins = set(map(tuple, state['coins']))
    others = {pos: name[:1].upper() for name, _, _, pos in state['others']}
    me = state['self'][3]
    expl = state['explosion_map']

    print('\n  board  (# wall, + crate, . free, o coin, 0-4 bomb timer, * fire, @ you)')
    for y in range(field.shape[1]):
        row = []
        for x in range(field.shape[0]):
            if (x, y) == me:            c = '@'
            elif (x, y) in bombs:       c = str(bombs[(x, y)])
            elif (x, y) in others:      c = others[(x, y)]
            elif expl[x, y] >= 1:       c = '*'
            elif (x, y) in coins:       c = 'o'
            elif field[x, y] == -1:     c = '#'
            elif field[x, y] == 1:      c = '+'
            else:                       c = '.'
            row.append(c)
        print('  ' + ' '.join(row))


def describe(state):
    print('=' * 72)
    print('game_state is a dict with these keys:\n')
    for key, value in state.items():
        if isinstance(value, np.ndarray):
            print(f"  {key:<15} ndarray{value.shape} of {value.dtype}, "
                  f"values seen: {sorted(set(value.flatten().tolist()))}")
        else:
            shown = repr(value)
            print(f"  {key:<15} {type(value).__name__:<8} {shown[:60]}")

    print(f"""
  self  is a 4-tuple:  (name, score, bomb_available, (x, y))
        -> name={state['self'][0]!r}  score={state['self'][1]}  """
          f"""bomb_available={state['self'][2]}  position={state['self'][3]}
  others is a list of the same tuple, one per living opponent
  bombs  is a list of ((x, y), countdown); countdown 0 means it detonates next step
  field  is -1 wall, 0 free, 1 crate  -- note field[x, y], x is the COLUMN
  explosion_map[x, y] >= 1 means fire is burning there right now""")
    draw(state)


def show_features(state, agent):
    print('\n' + '=' * 72)
    if agent == 'dqn_agent':
        from agent_code.dqn_agent.features import (
            state_to_features, CHANNEL_NAMES, SCALAR_NAMES)
        board, scalars = state_to_features(state)
        print(f'dqn_agent encodes that into board{board.shape} + scalars{scalars.shape}:\n')
        for i, name in enumerate(CHANNEL_NAMES):
            lit = int((board[i] > 0).sum())
            print(f"  plane {i}  {name:<11} {lit:>3} tiles lit, max {board[i].max():.2f}")
        print()
        for name, value in zip(SCALAR_NAMES, scalars):
            bar = '#' * int(value * 20)
            print(f"  {name:<16} {value:5.2f}  {bar}")
    else:
        from agent_code.my_agent.callbacks import state_to_features
        names = ['target_dir', 'target_kind', 'tile_here', 'tile_up', 'tile_right',
                 'tile_down', 'tile_left', 'bomb_ok', 'bomb_worth']
        feat = state_to_features(state)
        print(f'my_agent compresses that whole board into 9 numbers:\n\n  {feat}\n')
        for name, value in zip(names, feat):
            print(f"  {name:<13} {value}")
        print('\n  ^ this tuple is the *entire* key into the Q-table. Everything else '
              'about\n    the board above has been thrown away.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--agent', default='rule_based_agent')
    p.add_argument('--opponents', nargs='*', default=['rule_based_agent'] * 3)
    p.add_argument('--step', type=int, default=20, help='which step to freeze on')
    p.add_argument('--features', action='store_true',
                   help='also show how the agent encodes it')
    args = p.parse_args()

    world = make_world(args.agent, args.opponents)
    world.new_round()
    for _ in range(args.step):
        if not world.running:
            break
        world.do_step('WAIT')

    state = world.get_state_for_agent(world.agents[0])
    print(f'\nStep {state["step"]} of round {state["round"]}, '
          f'as seen by {state["self"][0]}')
    describe(state)
    if args.features:
        show_features(state, args.agent)
    world.end()


if __name__ == '__main__':
    main()
