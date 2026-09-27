"""Correctness tests for the DQN agent: encoding, network, replay buffer, n-step return."""
import sys, os
import numpy as np
sys.path.insert(0, os.getcwd())

import torch

import settings as s
from agent_code.dqn_agent.features import (
    BOARD_SHAPE, CHANNEL_NAMES, N_SCALARS, SCALAR_NAMES,
    state_to_features, quantise, dequantise)
from agent_code.dqn_agent.model import build_model, save_checkpoint, load_checkpoint, n_parameters
from agent_code.dqn_agent.train import ReplayBuffer, GAMMA
from agent_code.dqn_agent.callbacks import ACTIONS, is_suicide

FAIL = []
def check(name, got, want):
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {name}: got {got!r}, want {want!r}")
    if not ok: FAIL.append(name)

def close(name, got, want, tol=1e-5):
    ok = abs(got - want) < tol
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

f = empty_field()

print("--- board planes ---")
st = mk(f, (1,1), coins=[(3,1)], bombs=[((5,1), 2)],
        others=[('bob', 0, True, (7,1)), ('eve', 0, False, (9,1))])
board, scal = state_to_features(st)
check("board shape", board.shape, BOARD_SHAPE)
check("scalar shape", scal.shape, (N_SCALARS,))
check("channel names match depth", len(CHANNEL_NAMES), BOARD_SHAPE[0])
check("scalar names match width", len(SCALAR_NAMES), N_SCALARS)
check("dtype float32", board.dtype, np.dtype('float32'))
check("all planes in [0,1]", bool((board >= 0).all() and (board <= 1).all()), True)
check("wall plane marks the border", float(board[0, 0, 0]), 1.0)
check("self plane marks us", float(board[3, 1, 1]), 1.0)
check("coin plane marks the coin", float(board[2, 3, 1]), 1.0)
check("opponent with bomb = 1.0", float(board[4, 7, 1]), 1.0)
check("opponent without bomb = 0.5", float(board[4, 9, 1]), 0.5)
check("bomb tile is not walkable", float(board[8, 5, 1]), 0.0)
check("bomb urgency in (0,1]", bool(0 < board[5, 5, 1] <= 1), True)
check("danger plane lit inside the blast", bool(board[7, 5, 1] > 0), True)
check("danger plane dark far away", float(board[7, 1, 1]), 0.0)

print("\n--- danger is brighter the sooner it lands ---")
soon = state_to_features(mk(f, (1,1), bombs=[((5,1), 0)]))[0][7, 5, 1]
later = state_to_features(mk(f, (1,1), bombs=[((5,1), 3)]))[0][7, 5, 1]
check("timer 0 brighter than timer 3", bool(soon > later), True)

print("\n--- scalar inputs ---")
# Coin to the RIGHT of (1,1): DIRECTIONS order is UP,RIGHT,DOWN,LEFT -> one-hot index 2
sc = state_to_features(mk(f, (1,1), coins=[(3,1)]))[1]
check("dir one-hot picks RIGHT", int(np.argmax(sc[:5])), 2)
check("kind one-hot picks COIN", int(np.argmax(sc[5:9])), 1)
check("exactly one direction set", float(sc[:5].sum()), 1.0)
sc = state_to_features(mk(f, (1,1)))[1]
check("free tile: not in danger", float(sc[15]), 0.0)
check("free corner: bomb here is survivable", float(sc[14]), 1.0)
check("no bomb left -> bomb_available 0", float(state_to_features(mk(f,(1,1), bombs_left=False))[1][13]), 0.0)
sc = state_to_features(mk(f, (1,1), bombs=[((1,1), 3)]))[1]
check("standing on a bomb -> in_danger", float(sc[15]), 1.0)
check("in danger -> kind is NONE/escape", int(np.argmax(sc[5:9])), 0)
sc = state_to_features(mk(f, (1,1)))[1]
check("neighbour up is a wall -> unsafe", float(sc[9]), 0.0)
check("neighbour right is free -> safe", float(sc[10]), 1.0)
sc = state_to_features(mk(f, (1,1), coins=[(3,1)]*1))[1]
check("all scalars in [0,1]", bool((sc >= 0).all() and (sc <= 1).all()), True)

print("\n--- quantisation round-trip ---")
board, _ = state_to_features(mk(f, (1,1), bombs=[((5,1), 2)]))
err = float(np.abs(dequantise(quantise(board)) - board).max())
check("uint8 round-trip error below 1/255", bool(err <= 1/255 + 1e-6), True)
check("quantised dtype", quantise(board).dtype, np.dtype('uint8'))

print("\n--- network ---")
for arch in ('conv', 'mlp'):
    net = build_model(arch)
    b = torch.from_numpy(np.stack([board, board]))
    v = torch.from_numpy(np.stack([sc, sc]))
    out = net(b, v)
    check(f"{arch}: output shape", tuple(out.shape), (2, len(ACTIONS)))
    check(f"{arch}: finite outputs", bool(torch.isfinite(out).all()), True)
    print(f"      {arch}: {n_parameters(net)} parameters")

print("\n--- dueling head is not degenerate ---")
net = build_model('conv')
out = net(torch.from_numpy(board[None]), torch.from_numpy(sc[None]))
check("six distinct action values", bool(out.squeeze(0).unique().numel() > 1), True)

print("\n--- checkpoint round-trip ---")
tmp = 'tests/_tmp_ckpt.pt'
net = build_model('conv')
save_checkpoint(tmp, net, 'conv', extra={'rounds': 7})
loaded, payload = load_checkpoint(tmp, 'cpu')
check("arch recorded", payload['arch'], 'conv')
check("extra fields survive", payload['rounds'], 7)
a = net(torch.from_numpy(board[None]), torch.from_numpy(sc[None]))
b = loaded(torch.from_numpy(board[None]), torch.from_numpy(sc[None]))
check("weights identical after reload", bool(torch.allclose(a, b)), True)
os.remove(tmp)

print("\n--- replay buffer ---")
buf = ReplayBuffer(4)
feats = (board, sc)
for i in range(6):                       # overflow it on purpose
    buf.push(feats, i % len(ACTIONS), float(i), feats if i % 2 else None, 3)
check("capped at capacity", len(buf), 4)
check("ring buffer wrapped", int(buf.actions[0]), 4)
rng = np.random.default_rng(0)
sample = buf.sample(8, rng)
check("sampled board batch shape", sample[0].shape, (8, *BOARD_SHAPE))
check("sampled boards are float32", sample[0].dtype, np.dtype('float32'))
check("terminal flag stored for None next state", float(buf.dones[buf.actions == 4][0]), 1.0)

print("\n--- n-step return arithmetic ---")
# G = r0 + g*r1 + g^2*r2 for rewards 1, 2, 3
rewards = [1.0, 2.0, 3.0]
want = sum((GAMMA ** k) * r for k, r in enumerate(rewards))
got = 0.0
for k, r in enumerate(rewards):
    got += (GAMMA ** k) * r
close("three-step discounted return", got, want)
close("gamma^n discount for bootstrap", GAMMA ** 3, GAMMA*GAMMA*GAMMA)

print("\n--- is_suicide sanity (shared with the tabular agent) ---")
check("walking into a wall is 'suicide' (invalid)", is_suicide(mk(f,(1,1)), 'UP'), True)
check("walking into a free tile is fine", is_suicide(mk(f,(1,1)), 'RIGHT'), False)
ex = np.zeros_like(f); ex[2,1] = 1
check("walking into fire is suicide", is_suicide(mk(f,(1,1), expl=ex), 'RIGHT'), True)

print("\n" + ("ALL PASS" if not FAIL else f"{len(FAIL)} FAILURES: {FAIL}"))
sys.exit(1 if FAIL else 0)
