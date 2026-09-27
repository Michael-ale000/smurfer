"""
Inference code for a tabular Q-learning Bomberman agent.

The board is far too large to learn over directly (17x17 tiles with bombs, coins and
three opponents), so we compress every game state into a small hand-designed
*situation* tuple.  Two boards that look nothing alike but call for the same move map
to the same tuple, which is what makes tabular learning feasible here: the whole state
space is roughly 3e4 situations rather than astronomically many boards.

The tuple is deliberately egocentric and direction-based:

    target_dir   0..4  where to go (NONE / UP / RIGHT / DOWN / LEFT)
    target_kind  0..3  why we are going there (escape-or-none / coin / crate / enemy)
    tile_here    0..2  status of the tile under us
    tile_up      0..2  \\
    tile_right   0..2   |  status of the four neighbours
    tile_down    0..2   |
    tile_left    0..2  /
    bomb_ok      0..1  a bomb here would be both legal and survivable
    bomb_worth   0..2  what a bomb here would achieve (nothing / crates / an enemy)

Tile status is 0 = free and safe, 1 = blocked, 2 = free but on fire or about to be.

When we are standing in a blast radius, ``target_dir`` switches to pointing at the
nearest survivable tile.  That way the same field carries "where do I want to go"
in both the calm and the panicking case, and ``tile_here == 2`` tells the agent
which of the two it is looking at.
"""

import logging
import os
import pickle
import random

import numpy as np

from .game_utils import (
    DIRECTIONS,
    LARGE,
    bfs_first_step,
    bfs_to_safety,
    bomb_deadlines,
    can_survive,
    crates_hit,
    is_lethal_at,
    opponents_hit,
    passable,
)

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

# The framework chdir's into this directory before calling us, but deriving the path
# from __file__ keeps the agent working no matter who imports it.
MODEL_FILE = os.path.join(os.path.dirname(__file__), 'q_table.pt')

# Tile status codes
TILE_FREE, TILE_BLOCKED, TILE_DANGER = 0, 1, 2

# Target kinds
KIND_NONE, KIND_COIN, KIND_CRATE, KIND_ENEMY = 0, 1, 2, 3


def setup(self):
    """Load the Q-table, or start an empty one when training from scratch."""
    self.rng = np.random.default_rng()

    # Per-step DEBUG logging writes several lines per step, which over a
    # million-step training run costs more disk than the Q-table itself. Bulk
    # training turns it down to INFO, which keeps the one-line-per-round summary
    # that the training curves are read from. Never touches tournament play.
    if os.environ.get('MY_AGENT_QUIET'):
        self.logger.setLevel(logging.INFO)

    if os.path.isfile(MODEL_FILE):
        with open(MODEL_FILE, 'rb') as f:
            self.q_table = pickle.load(f)
        self.logger.info(f'Loaded Q-table with {len(self.q_table)} situations.')
    else:
        if not self.train:
            self.logger.warning('No Q-table found -- the agent will act randomly.')
        self.q_table = {}
        self.logger.info('Starting from an empty Q-table.')


def q_values(self, features):
    """Q-row for a situation, created lazily so unseen situations cost nothing."""
    if features not in self.q_table:
        self.q_table[features] = np.zeros(len(ACTIONS), dtype=np.float32)
    return self.q_table[features]


def act(self, game_state: dict) -> str:
    """Pick a move: epsilon-greedy while training, greedy in a tournament."""
    features = state_to_features(game_state)

    epsilon = getattr(self, 'epsilon', 0.0) if self.train else 0.0
    if self.train and random.random() < epsilon:
        self.logger.debug('Exploring.')
        return explore_action(self, game_state, features)

    q = q_values(self, features)
    # Break ties at random so the agent does not get stuck facing one direction.
    best = np.flatnonzero(q == q.max())
    action = ACTIONS[self.rng.choice(best)]
    self.logger.debug(f'Situation {features} -> {action} (Q={q.max():.2f})')
    return action


def explore_action(self, game_state, features):
    """Exploration that is biased away from obviously fatal moves.

    Purely uniform exploration spends most of its time walking into fire or dropping
    unescapable bombs, which floods the replay buffer with deaths and teaches very
    little.  We keep the randomness but drop the moves we already know are suicide.
    """
    safe = [a for a in ACTIONS if not is_suicide(game_state, a)]
    pool = safe if safe else ACTIONS
    # Bombs are rare but high-value; keep them in the mix without dominating it.
    weights = np.array([0.4 if a == 'BOMB' else 1.0 for a in pool])
    return str(self.rng.choice(pool, p=weights / weights.sum()))


def is_suicide(game_state, action):
    """Cheap lookahead: would this single move put us somewhere we cannot survive?"""
    x, y = game_state['self'][3]
    deadlines = bomb_deadlines(game_state)

    if action == 'BOMB':
        if not game_state['self'][2]:
            return True
        with_bomb = bomb_deadlines(game_state, extra_bomb=(x, y))
        return not can_survive(game_state, (x, y), with_bomb, extra_wait_first=True)

    if action == 'WAIT':
        nx, ny = x, y
    else:
        dx, dy = DIRECTIONS[ACTIONS.index(action)]
        nx, ny = x + dx, y + dy
        if not passable(game_state, nx, ny):
            return True

    if is_lethal_at(deadlines, nx, ny, 1):
        return True
    return not can_survive(game_state, (nx, ny), deadlines)


def tile_status(game_state, deadlines, x, y, is_own_tile=False):
    """Compress a tile into free / blocked / dangerous."""
    if not is_own_tile and not passable(game_state, x, y):
        return TILE_BLOCKED
    # "Dangerous" means fire is coming, whether this move or a few moves out.
    if deadlines[x, y] < LARGE:
        return TILE_DANGER
    return TILE_FREE


def state_to_features(game_state: dict):
    """Compress a game state into the situation tuple described in the module docstring."""
    if game_state is None:
        return None

    field = game_state['field']
    x, y = game_state['self'][3]
    bombs_left = game_state['self'][2]
    coins = game_state['coins']
    others = [pos for _, _, _, pos in game_state['others']]

    deadlines = bomb_deadlines(game_state)
    in_danger = deadlines[x, y] < LARGE

    # --- where do we want to go, and why -------------------------------------
    if in_danger:
        # Survival overrides every other objective.
        direction = bfs_to_safety(game_state, (x, y), deadlines)
        kind = KIND_NONE
    else:
        direction = bfs_first_step(game_state, (x, y), coins, avoid_deadlines=deadlines)
        kind = KIND_COIN
        if direction is None:
            crates = [tuple(c) for c in np.argwhere(field == 1)]
            direction = bfs_first_step(game_state, (x, y), crates, avoid_deadlines=deadlines)
            kind = KIND_CRATE
        if direction is None:
            direction = bfs_first_step(game_state, (x, y), others, avoid_deadlines=deadlines)
            kind = KIND_ENEMY
        if direction is None:
            kind = KIND_NONE

    target_dir = 0 if direction is None else direction + 1

    # --- what is around us ----------------------------------------------------
    tile_here = tile_status(game_state, deadlines, x, y, is_own_tile=True)
    neighbours = tuple(
        tile_status(game_state, deadlines, x + dx, y + dy)
        for dx, dy in DIRECTIONS
    )

    # --- is a bomb here a good idea -------------------------------------------
    if bombs_left:
        with_bomb = bomb_deadlines(game_state, extra_bomb=(x, y))
        bomb_ok = int(can_survive(game_state, (x, y), with_bomb, extra_wait_first=True))
    else:
        bomb_ok = 0

    if opponents_hit(game_state, x, y) > 0:
        bomb_worth = 2
    elif crates_hit(field, x, y) > 0:
        bomb_worth = 1
    else:
        bomb_worth = 0

    return (target_dir, kind, tile_here) + neighbours + (bomb_ok, bomb_worth)
