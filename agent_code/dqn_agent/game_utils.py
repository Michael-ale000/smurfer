"""
Pure game-logic helpers shared by callbacks.py (inference) and train.py (learning).

Nothing in here learns anything -- these are exact simulations of the rules in
environment.py, used to build the feature vector.  Keeping them in one place means
the features seen during training are bit-for-bit the features seen in a tournament.

Timing model (derived from environment.py -> do_step/update_bombs/update_explosions)
-----------------------------------------------------------------------------------
A step runs: agents act -> collect coins -> update explosions -> update bombs
-> evaluate explosions.  So the consequence of a bomb is felt *after* the move
we are about to make.  We count our own moves from 1.

  * A bomb observed with countdown ``t`` detonates right after our move ``t + 1``.
    Its blast is deadly at the end of move ``t + 1`` and again at ``t + 2``
    (Explosion is created with timer 2 and stays at stage 0 for two evaluations).
  * ``explosion_map[x, y] >= 1`` means an explosion is already burning and will
    kill us at the end of move 1 -- but not afterwards.
  * Dropping a bomb costs move 1 and the blast lands after move 5, i.e. we get
    four moves to run.

We call ``t + 1`` the *deadline* of a tile: the first of our moves at whose end
standing there is fatal.
"""

import numpy as np

import settings as s

# (dx, dy) in the same order as ACTIONS[:4] in callbacks.py
DIRECTIONS = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # UP, RIGHT, DOWN, LEFT

# How long a bomb's fire lingers, in our own move counting.  A blast that lands
# after move d is still lethal after move d + 1.
BLAST_LINGER = 1

# Moves between dropping a bomb and its blast landing (see module docstring).
OWN_BOMB_DEADLINE = s.BOMB_TIMER + 1

# Stand-in for "no bomb threatens this tile"; larger than any reachable deadline.
LARGE = 999


def blast_coords(field, x, y, power=s.BOMB_POWER):
    """Tiles hit by a bomb at (x, y).  Stone walls stop the blast, crates do not."""
    coords = [(x, y)] # blasting coordinate tuple.
    for dx, dy in DIRECTIONS:
        for i in range(1, power + 1):
            nx, ny = x + i * dx, y + i * dy
            if field[nx, ny] == -1:
                break
            coords.append((nx, ny))
    return coords


def bomb_deadlines(game_state, extra_bomb=None):
    """Map every tile to the earliest of our moves at whose end it turns lethal.

    Returns an int array where ``LARGE`` means "no bomb threatens this tile".
    ``extra_bomb`` lets us ask "what if I dropped a bomb at (x, y) right now?"
    """
    field = game_state['field']
    deadlines = np.full(field.shape, LARGE, dtype=int) #initialize all the field value with 999

    bombs = list(game_state['bombs'])
    if extra_bomb is not None:
        # A bomb we are considering dropping this very move.
        bombs = bombs + [(extra_bomb, s.BOMB_TIMER)]

    for (bx, by), timer in bombs:
        deadline = timer + 1
        for (cx, cy) in blast_coords(field, bx, by):
            if deadline < deadlines[cx, cy]:
                deadlines[cx, cy] = deadline

    # An explosion already burning kills us at the end of move 1 only.
    burning = game_state['explosion_map'] >= 1
    deadlines[burning] = np.minimum(deadlines[burning], 1)

    return deadlines


def is_lethal_at(deadlines, x, y, move):
    """Would standing on (x, y) at the end of our move number ``move`` kill us?"""
    d = deadlines[x, y]
    return d <= move <= d + BLAST_LINGER


def passable(game_state, x, y):
    """Can we walk onto this tile?  Walls, crates, bombs and agents all block."""
    if game_state['field'][x, y] != 0:
        return False
    if any((bx, by) == (x, y) for (bx, by), _ in game_state['bombs']):
        return False
    if any(pos == (x, y) for _, _, _, pos in game_state['others']):
        return False
    return True


def free_tiles(game_state):
    """Boolean array of tiles we could in principle stand on (ignores fire)."""
    field = game_state['field']
    free = field == 0
    for (bx, by), _ in game_state['bombs']:
        free[bx, by] = False
    for _, _, _, (ox, oy) in game_state['others']:
        free[ox, oy] = False
    return free


def can_survive(game_state, start, deadlines, extra_wait_first=False):
    """Is there any sequence of moves from ``start`` that outlives every active bomb?

    A time-indexed breadth-first search over (position, move number).  We only need
    to look as far as the last deadline plus the lingering fire, which is a handful
    of steps, so this stays far inside the 0.5 s budget.

    ``extra_wait_first`` models the BOMB action: it consumes move 1 without moving.
    """
    horizon = int(deadlines[deadlines < LARGE].max()) + BLAST_LINGER if (deadlines < LARGE).any() else 0
    if horizon == 0:
        return True

    free = free_tiles(game_state)
    # We are standing on our own tile, so it is available to us regardless.
    free[start] = True

    frontier = {start}
    move = 1

    if extra_wait_first:
        # Move 1 is spent dropping the bomb; we stay where we are.
        if is_lethal_at(deadlines, start[0], start[1], 1):
            return False
        move = 2

    while move <= horizon:
        nxt = set()
        for (x, y) in frontier:
            for dx, dy in DIRECTIONS + [(0, 0)]:
                nx, ny = x + dx, y + dy
                if not free[nx, ny]:
                    continue
                if is_lethal_at(deadlines, nx, ny, move):
                    continue
                nxt.add((nx, ny))
        if not nxt:
            return False
        frontier = nxt
        move += 1

    return True


def bfs_first_step(game_state, start, targets, avoid_deadlines=None):
    """Direction index (0..3) of the first step on a shortest path to any target.

    Returns ``None`` when no target is reachable.  When ``avoid_deadlines`` is given,
    tiles that are lethal on arrival are treated as walls, so the agent never routes
    itself through fire to reach a coin.
    """
    if not targets:
        return None

    target_set = set(targets)
    if start in target_set:
        return None

    free = free_tiles(game_state)
    # Crates and opponents are legal destinations even though we cannot enter them.
    for t in target_set:
        free[t] = True
    free[start] = True

    # (position, direction we first stepped in, distance)
    frontier = [(start, None, 0)]
    seen = {start}
    head = 0

    while head < len(frontier):
        (x, y), first_dir, dist = frontier[head]
        head += 1
        for idx, (dx, dy) in enumerate(DIRECTIONS):
            nx, ny = x + dx, y + dy
            if (nx, ny) in seen or not free[nx, ny]:
                continue
            if avoid_deadlines is not None and is_lethal_at(avoid_deadlines, nx, ny, dist + 1):
                continue
            step_dir = idx if first_dir is None else first_dir
            if (nx, ny) in target_set:
                return step_dir
            seen.add((nx, ny))
            frontier.append(((nx, ny), step_dir, dist + 1))

    return None


def bfs_to_safety(game_state, start, deadlines):
    """Direction index (0..3) of the first step toward the nearest survivable tile."""
    free = free_tiles(game_state)
    free[start] = True

    frontier = [(start, None, 0)]
    seen = {start}
    head = 0

    while head < len(frontier):
        (x, y), first_dir, dist = frontier[head]
        head += 1
        for idx, (dx, dy) in enumerate(DIRECTIONS):
            nx, ny = x + dx, y + dy
            if (nx, ny) in seen or not free[nx, ny]:
                continue
            move = dist + 1
            if is_lethal_at(deadlines, nx, ny, move):
                continue
            step_dir = idx if first_dir is None else first_dir
            if deadlines[nx, ny] >= LARGE:
                return step_dir
            seen.add((nx, ny))
            frontier.append(((nx, ny), step_dir, move))

    return None


def crates_hit(field, x, y):
    """How many crates a bomb dropped at (x, y) would destroy."""
    return sum(1 for (cx, cy) in blast_coords(field, x, y) if field[cx, cy] == 1)


def opponents_hit(game_state, x, y):
    """How many opponents currently stand in the blast of a bomb dropped at (x, y)."""
    coords = set(blast_coords(game_state['field'], x, y))
    return sum(1 for _, _, _, pos in game_state['others'] if pos in coords)
