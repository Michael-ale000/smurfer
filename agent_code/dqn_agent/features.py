"""
State encoding for the deep agent.

The tabular agent had to throw almost everything away: its 9-field tuple exists only
because a table needs a small, discrete key.  A network does not, so here the state
stays *spatial*.  Every step is encoded as

    board    (C, 17, 17) float32 in [0, 1] -- one plane per kind of thing on the map
    scalars  (N,)        float32 in [0, 1] -- a few global facts plus the BFS hints

and the network sees both.  Keeping the board raw is the point of the exercise: the
convolution can learn "a crate two tiles away with a corner to duck behind" without
anyone hand-coding that concept, which is exactly what the tuple could not express.

The scalars are not raw.  They carry the same BFS target direction and bomb-safety
verdict the tabular agent used, because those come from an exact simulation of the
rules (``game_utils``) that a network would otherwise have to rediscover from reward
alone.  Handing them over costs 21 inputs and saves a great deal of training.

All values are in [0, 1] so the replay buffer can store boards as uint8 (a factor of
four less memory, which is what makes a 40k-transition buffer fit comfortably in RAM).
"""

import numpy as np

import settings as s

from .game_utils import (
    DIRECTIONS,
    LARGE,
    OWN_BOMB_DEADLINE,
    BLAST_LINGER,
    bfs_first_step,
    bfs_to_safety,
    bomb_deadlines,
    can_survive,
    crates_hit,
    is_lethal_at,
    opponents_hit,
    passable,
)

# --- board planes ------------------------------------------------------------
CHANNEL_NAMES = [
    'wall',        # stone walls #static map
    'crate',       # destructible crates #static map
    'coin',        # collectable coins # moving objects
    'self',        # us #moving objects
    'other',       # opponents (1.0 if they still hold a bomb, 0.5 if not) #moving objects
    'bomb',        # bombs, scaled by urgency (1.0 = about to go off) #threat layers
    'explosion',   # fire burning right now #threat layers
    'danger',      # how soon each tile turns lethal (1.0 = this move) #threat layers
    'walkable',    # tiles we could step onto at all #threat layers
]
N_CHANNELS = len(CHANNEL_NAMES) # planes in three groups(static map, moving objects and threat layers)
BOARD_SHAPE = (N_CHANNELS, s.COLS, s.ROWS) #spatial, one picture per state or thing(what is where)

# The longest deadline that can exist: our own freshly dropped bomb, plus lingering fire.
MAX_DEADLINE = OWN_BOMB_DEADLINE + BLAST_LINGER

# --- scalar inputs -----------------------------------------------------------
SCALAR_NAMES = [
    'dir_none', 'dir_up', 'dir_right', 'dir_down', 'dir_left',   # BFS target, one-hot, which way to go
    'kind_none', 'kind_coin', 'kind_crate', 'kind_enemy',        # why we are going there
    'safe_up', 'safe_right', 'safe_down', 'safe_left',           # neighbour is free and not on fire
    'bomb_available', 'bomb_survivable', 'in_danger',
    'crates_in_blast', 'enemies_in_blast',                       # what a bomb here would achieve
    'step', 'coins_visible', 'opponents_left',                   #global clock and score context
]
N_SCALARS = len(SCALAR_NAMES) #21 (global facts + derived hints) (what should i do about it) (divided into 5 groups)


def state_to_features(game_state: dict):
    """Encode one game state as ``(board, scalars)``.  ``None`` in, ``None`` out."""
    if game_state is None:
        return None

    deadlines = bomb_deadlines(game_state)
    return board_planes(game_state, deadlines), scalar_inputs(game_state, deadlines)


def board_planes(game_state, deadlines):
    """The (C, 17, 17) stack described in ``CHANNEL_NAMES``."""
    field = game_state['field']
    board = np.zeros(BOARD_SHAPE, dtype=np.float32)

    board[0] = (field == -1)
    board[1] = (field == 1)

    for (cx, cy) in game_state['coins']:
        board[2, cx, cy] = 1.0

    x, y = game_state['self'][3]
    board[3, x, y] = 1.0

    for _, _, bombs_left, (ox, oy) in game_state['others']:
        # An opponent that still has a bomb is a very different threat from one that
        # does not, and that fits in the same plane as an intensity.
        board[4, ox, oy] = 1.0 if bombs_left else 0.5

    for (bx, by), timer in game_state['bombs']:
        board[5, bx, by] = (s.BOMB_TIMER - timer + 1) / (s.BOMB_TIMER + 1)

    board[6] = np.clip(game_state['explosion_map'] / s.EXPLOSION_TIMER, 0.0, 1.0)

    # Danger as "how soon", so nearer deadlines are brighter than distant ones.
    threatened = deadlines < LARGE
    board[7][threatened] = (MAX_DEADLINE + 1 - np.clip(deadlines[threatened], 1, MAX_DEADLINE)) / MAX_DEADLINE

    walkable = field == 0
    for (bx, by), _ in game_state['bombs']:
        walkable[bx, by] = False
    for _, _, _, (ox, oy) in game_state['others']:
        walkable[ox, oy] = False
    board[8] = walkable

    return board


def scalar_inputs(game_state, deadlines):
    """Global facts and the BFS hints, all scaled into [0, 1]."""
    field = game_state['field']
    x, y = game_state['self'][3]
    bombs_left = game_state['self'][2]
    others = [pos for _, _, _, pos in game_state['others']]

    in_danger = bool(deadlines[x, y] < LARGE)

    # Same objective ladder as the tabular agent: survive, else coin, else crate,
    # else hunt.  Only the *first step* of the path is exposed, which is all a
    # one-step policy can act on anyway.
    if in_danger:
        direction = bfs_to_safety(game_state, (x, y), deadlines)
        kind = 0
    else:
        direction = bfs_first_step(game_state, (x, y), game_state['coins'],
                                   avoid_deadlines=deadlines)
        kind = 1
        if direction is None:
            crates = [tuple(c) for c in np.argwhere(field == 1)]
            direction = bfs_first_step(game_state, (x, y), crates, avoid_deadlines=deadlines)
            kind = 2
        if direction is None:
            direction = bfs_first_step(game_state, (x, y), others, avoid_deadlines=deadlines)
            kind = 3
        if direction is None:
            kind = 0

    out = np.zeros(N_SCALARS, dtype=np.float32)
    out[0 if direction is None else 1 + direction] = 1.0
    out[5 + kind] = 1.0

    for i, (dx, dy) in enumerate(DIRECTIONS):
        nx, ny = x + dx, y + dy
        safe = passable(game_state, nx, ny) and not is_lethal_at(deadlines, nx, ny, 1)
        out[9 + i] = float(safe)

    out[13] = float(bombs_left)
    if bombs_left:
        with_bomb = bomb_deadlines(game_state, extra_bomb=(x, y))
        out[14] = float(can_survive(game_state, (x, y), with_bomb, extra_wait_first=True))
    out[15] = float(in_danger)

    # Blast payoff, normalised by a generous upper bound rather than the true maximum:
    # the exact scale does not matter, only that it stays in [0, 1] and stays monotone.
    out[16] = min(crates_hit(field, x, y), 6) / 6.0
    out[17] = min(opponents_hit(game_state, x, y), 3) / 3.0

    out[18] = min(game_state['step'], s.MAX_STEPS) / s.MAX_STEPS
    out[19] = min(len(game_state['coins']), 9) / 9.0
    out[20] = len(others) / 3.0

    return out


def quantise(board):
    """Board -> uint8, for the replay buffer.  Every plane is already in [0, 1]."""
    return np.clip(board * 255.0, 0, 255).astype(np.uint8)


def dequantise(board_u8):
    return board_u8.astype(np.float32) / 255.0
