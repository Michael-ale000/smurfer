"""Correctness tests for the feature encoding and the bomb-timing model."""
import sys, os
import numpy as np
sys.path.insert(0, os.getcwd())

import settings as s
from agent_code.my_agent.callbacks import state_to_features, is_suicide, ACTIONS
from agent_code.my_agent.game_utils import (
    bomb_deadlines, can_survive, blast_coords, LARGE, is_lethal_at)

FAIL = []
def check(name, got, want):
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {name}: got {got!r}, want {want!r}")
    if not ok: FAIL.append(name)

def empty_field():
    f = np.zeros((s.COLS, s.ROWS), dtype=int)
    f[0,:] = f[-1,:] = f[:,0] = f[:,-1] = -1
    for x in range(1, s.COLS-1):
        for y in range(1, s.ROWS-1):
            if x % 2 == 0 and y % 2 == 0:
                f[x, y] = -1
    return f

def mk(field, pos, bombs=(), coins=(), others=(), bombs_left=True, expl=None):
    return {'round':1,'step':10,'field':field,'bombs':list(bombs),'coins':list(coins),
            'self':('me',0,bombs_left,pos),'others':list(others),
            'explosion_map': np.zeros_like(field) if expl is None else expl,
            'user_input':None}

print("--- direction encoding (ACTIONS[:4] = UP,RIGHT,DOWN,LEFT) ---")
f = empty_field()
# Coin to the RIGHT of (1,1) -> RIGHT is index 1 -> target_dir 2
st = mk(f, (1,1), coins=[(3,1)])
check("coin to the right -> target_dir=RIGHT(2)", state_to_features(st)[0], 2)
# Coin BELOW (image coords: +y is down) -> DOWN index 2 -> target_dir 3
st = mk(f, (1,1), coins=[(1,3)])
check("coin below -> target_dir=DOWN(3)", state_to_features(st)[0], 3)
# Coin ABOVE
st = mk(f, (1,3), coins=[(1,1)])
check("coin above -> target_dir=UP(1)", state_to_features(st)[0], 1)
# Coin LEFT
st = mk(f, (3,1), coins=[(1,1)])
check("coin left -> target_dir=LEFT(4)", state_to_features(st)[0], 4)
check("target kind = coin", state_to_features(st)[1], 1)

print("\n--- bomb timing model ---")
# A bomb with timer t detonates after our move t+1, lethal at moves t+1 and t+2.
st = mk(f, (1,1), bombs=[((1,1), 0)])
d = bomb_deadlines(st)
check("timer 0 -> deadline 1", int(d[1,1]), 1)
check("lethal at move 1", is_lethal_at(d,1,1,1), True)
check("still lethal at move 2 (fire lingers)", is_lethal_at(d,1,1,2), True)
check("safe at move 3", is_lethal_at(d,1,1,3), False)
st = mk(f, (1,1), bombs=[((1,1), 3)])
check("fresh bomb timer 3 -> deadline 4", int(bomb_deadlines(st)[1,1]), 4)
# Explosion already burning
ex = np.zeros_like(f); ex[1,1] = 1
st = mk(f, (1,2), expl=ex)
check("burning explosion -> deadline 1", int(bomb_deadlines(st)[1,1]), 1)

print("\n--- blast geometry ---")
# Walls sit at even-x AND even-y, so (2,2) is a wall: a bomb at (1,2) is stopped there.
bc2 = blast_coords(f, 1, 2)
check("blast stopped by wall at (2,2)", (3,2) in bc2, False)
check("wall tile itself not in blast", (2,2) in bc2, False)
bc = blast_coords(f, 1, 1)
check("blast covers own tile", (1,1) in bc, True)
check("blast reaches 3 right", (4,1) in bc, True)
check("blast reaches 3 down along free column", (1,4) in bc, True)

print("\n--- survival / escape reasoning ---")
# Dead-end corridor: bomb at the closed end, agent one tile in -> must be able to run out
st = mk(f, (1,1), bombs=[((1,1), 3)])
check("agent on fresh bomb in open corner can escape", can_survive(st, (1,1), bomb_deadlines(st)), True)
# Build a true dead end: (1,1)..(1,3) corridor closed off
g = empty_field()
g[2,1] = 1; g[2,3] = 1   # crates seal the side exits
st = mk(g, (1,1), bombs=[((1,3), 3)])
check("trapped in sealed dead end -> cannot survive", can_survive(st, (1,1), bomb_deadlines(st)), False)

print("\n--- bomb_ok (would dropping here be survivable?) ---")
st = mk(f, (1,1))
feat = state_to_features(st)
check("open corner: bomb_ok = 1", feat[7], 1)
g = empty_field()
g[2,1] = 1; g[1,2] = 1   # box ourselves into a 1-tile pocket
st = mk(g, (1,1))
check("sealed pocket: bomb_ok = 0", state_to_features(st)[7], 0)
check("sealed pocket: is_suicide('BOMB')", is_suicide(st, 'BOMB'), True)
check("no bomb available -> bomb_ok = 0", state_to_features(mk(f,(1,1),bombs_left=False))[7], 0)

print("\n--- bomb_worth ---")
g = empty_field(); g[2,1] = 1
check("crate in blast -> bomb_worth=1", state_to_features(mk(g,(1,1)))[8], 1)
st = mk(f, (1,1), others=[('o',0,True,(3,1))])
check("enemy in blast -> bomb_worth=2", state_to_features(st)[8], 2)
check("nothing in blast -> bomb_worth=0", state_to_features(mk(f,(1,1)))[8], 0)

print("\n--- danger overrides target ---")
st = mk(f, (1,1), bombs=[((1,1), 3)], coins=[(5,1)])
feat = state_to_features(st)
check("in danger -> tile_here=2", feat[2], 2)
check("in danger -> kind=NONE(0)", feat[1], 0)
check("in danger -> points at an escape", feat[0] != 0, True)

print("\n--- is_suicide sanity ---")
st = mk(f, (1,1))
check("walking into a wall is 'suicide' (invalid)", is_suicide(st, 'UP'), True)
check("walking into free tile is fine", is_suicide(st, 'RIGHT'), False)
ex = np.zeros_like(f); ex[2,1] = 1
st = mk(f, (1,1), expl=ex)
check("walking into fire is suicide", is_suicide(st, 'RIGHT'), True)

print("\n" + ("ALL PASS" if not FAIL else f"{len(FAIL)} FAILURES: {FAIL}"))
sys.exit(1 if FAIL else 0)
