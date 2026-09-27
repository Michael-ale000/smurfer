"""
The network, and the checkpoint format that carries it between processes.

Two torsos are provided and selected with ``DQN_ARCH``:

``conv`` (default)
    Three convolutions over the (C, 17, 17) board, striding down to 5x5, then a fully
    connected head.  Convolution is the right prior here: "am I in a blast lane", "is
    there a crate next to a corner" are local patterns that mean the same thing
    wherever they occur on the map, so the weights should be shared across positions.

    The widths are deliberately small.  Training is CPU-bound on a laptop and the cost
    of a gradient step is what decides how many rounds fit in an evening: measured at
    batch 32, (16, 32, 64) costs ~12 ms per update against ~18 ms for (32, 64, 64),
    for no measurable difference in what the agent learns on a 17x17 board.  Override
    with ``DQN_CONV_CHANNELS=32,64,64`` and ``DQN_HIDDEN=256`` if you want to compare.

``mlp``
    The same input flattened into one vector.  Included as the ablation -- it has more
    parameters in its first layer than the whole conv torso and still generalises worse,
    which is the cleanest way to show that the spatial structure is doing real work.

Both end in a **dueling** head: the torso output is split into a single state value
V(s) and per-action advantages A(s, a), recombined as

    Q(s, a) = V(s) + A(s, a) - mean_a A(s, a)

In Bomberman most states are dominated by "how dangerous is it here", which is a
property of the state and not of the action; the dueling split lets one output learn
that once instead of six outputs learning it six times.
"""

import os

import torch
import torch.nn as nn

from .features import BOARD_SHAPE, N_SCALARS

N_ACTIONS = 6
SCALAR_WIDTH = 64
HIDDEN = int(os.environ.get('DQN_HIDDEN', 256))
CONV_CHANNELS = tuple(int(c) for c in os.environ.get('DQN_CONV_CHANNELS', '16,32,64').split(','))


class _DuelingHead(nn.Module):
    def __init__(self, in_features, n_actions):
        super().__init__()
        self.value = nn.Linear(in_features, 1)
        self.advantage = nn.Linear(in_features, n_actions)

    def forward(self, h):
        v = self.value(h)
        a = self.advantage(h)
        return v + a - a.mean(dim=1, keepdim=True)


class ConvQNet(nn.Module):
    """Convolutional torso over the board, with the scalars merged in at the head."""

    def __init__(self, in_channels=BOARD_SHAPE[0], n_scalars=N_SCALARS, n_actions=N_ACTIONS):
        super().__init__()
        c1, c2, c3 = CONV_CHANNELS
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, c1, 3, stride=1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(c1, c2, 3, stride=2, padding=1), nn.ReLU(inplace=True),   # 17 -> 9
            nn.Conv2d(c2, c3, 3, stride=2, padding=1), nn.ReLU(inplace=True),   # 9  -> 5
        )
        conv_out = c3 * 5 * 5
        self.scalar_fc = nn.Sequential(nn.Linear(n_scalars, SCALAR_WIDTH), nn.ReLU(inplace=True))
        self.trunk = nn.Sequential(nn.Linear(conv_out + SCALAR_WIDTH, HIDDEN), nn.ReLU(inplace=True))
        self.head = _DuelingHead(HIDDEN, n_actions)

    def forward(self, board, scalars):
        h = self.conv(board).flatten(1)
        h = torch.cat([h, self.scalar_fc(scalars)], dim=1)
        return self.head(self.trunk(h))


class MLPQNet(nn.Module):
    """Ablation torso: the same information, flattened, no spatial weight sharing."""

    def __init__(self, in_channels=BOARD_SHAPE[0], n_scalars=N_SCALARS, n_actions=N_ACTIONS):
        super().__init__()
        flat = in_channels * BOARD_SHAPE[1] * BOARD_SHAPE[2] + n_scalars
        self.trunk = nn.Sequential(
            nn.Linear(flat, 512), nn.ReLU(inplace=True),
            nn.Linear(512, HIDDEN), nn.ReLU(inplace=True),
        )
        self.head = _DuelingHead(HIDDEN, n_actions)

    def forward(self, board, scalars):
        h = torch.cat([board.flatten(1), scalars], dim=1)
        return self.head(self.trunk(h))


ARCHITECTURES = {'conv': ConvQNet, 'mlp': MLPQNet}


def build_model(arch='conv', in_channels=BOARD_SHAPE[0], n_scalars=N_SCALARS,
                n_actions=N_ACTIONS):
    if arch not in ARCHITECTURES:
        raise ValueError(f'Unknown DQN_ARCH {arch!r}; choose from {sorted(ARCHITECTURES)}')
    return ARCHITECTURES[arch](in_channels, n_scalars, n_actions)


def n_parameters(model):
    return sum(p.numel() for p in model.parameters())


def save_checkpoint(path, model, arch, extra=None):
    """Write a checkpoint that ``load_checkpoint`` can rebuild without any guessing.

    The architecture and input shapes travel with the weights, so a tournament run
    never has to be told which variant produced the file.
    """
    payload = {
        'arch': arch,
        'in_channels': BOARD_SHAPE[0],
        'n_scalars': N_SCALARS,
        'n_actions': N_ACTIONS,
        'state_dict': model.state_dict(),
    }
    payload.update(extra or {})
    torch.save(payload, path)


def load_checkpoint(path, device='cpu'):
    """Rebuild the model from a checkpoint.  Returns ``(model, payload)``."""
    payload = torch.load(path, map_location=device, weights_only=False)
    model = build_model(payload.get('arch', 'conv'),
                        payload.get('in_channels', BOARD_SHAPE[0]),
                        payload.get('n_scalars', N_SCALARS),
                        payload.get('n_actions', N_ACTIONS))
    model.load_state_dict(payload['state_dict'])
    model.to(device)
    return model, payload
