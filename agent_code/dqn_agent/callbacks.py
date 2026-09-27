"""
Inference code for the deep Q-network agent.

Loaded in every game, with or without ``--train``.  Everything that only matters while
learning lives in ``train.py``; what happens here is: encode the state, run one forward
pass, take the argmax.  A single forward pass of the conv net is well under a
millisecond on CPU, so the 0.5 s per-step budget is never in question.

Exploration lives here too, because ``act`` is the only place the framework lets an
agent choose a move, and because the exploration policy is deliberately not uniform --
see ``explore_action``.
"""

import logging
import os

import numpy as np
import torch

from .features import state_to_features
from .game_utils import (
    DIRECTIONS,
    bomb_deadlines,
    can_survive,
    is_lethal_at,
    passable,
)
from .model import build_model, load_checkpoint, n_parameters

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

# The framework chdir's into this directory before calling us, but deriving the path
# from __file__ keeps the agent working no matter who imports it.
MODEL_FILE = os.path.join(os.path.dirname(__file__), 'dqn_model.pt')
HISTORY_FILE = os.path.join(os.path.dirname(__file__), 'training_history.pt')


def pick_device():
    """CPU unless told otherwise.

    The network is tiny.  On a small net with batch 64 the per-kernel launch overhead
    of a GPU backend (MPS especially) is larger than the arithmetic it saves, so CPU is
    usually the *faster* choice here -- but the override is there to measure it:

        DQN_DEVICE=mps python train_dqn.py     # or cuda, or cpu
    """
    requested = os.environ.get('DQN_DEVICE', 'auto')
    if requested != 'auto':
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def setup(self):
    """Load the network, or start an untrained one when training from scratch."""
    self.rng = np.random.default_rng()

    # Per-step DEBUG logging writes several lines per step, which over a million-step
    # training run costs more disk than the model itself.  Bulk training turns it down
    # to INFO, which keeps the one-line-per-round summary the curves are read from.
    if os.environ.get('DQN_QUIET'):
        self.logger.setLevel(logging.INFO)

    # The tournament assumes exclusive access to *one* thread of the machine, so a
    # tournament game must not quietly spin up eight. Training is a different matter
    # and gets whatever torch chooses, unless DQN_THREADS says otherwise.
    threads = os.environ.get('DQN_THREADS')
    if threads:
        torch.set_num_threads(int(threads))
    elif not self.train:
        torch.set_num_threads(1)

    self.device = pick_device()
    self.arch = os.environ.get('DQN_ARCH', 'conv')
    # Filtering out moves the rules say are fatal, on top of the learned policy.
    # Off by default: an evaluation should measure the policy, not the filter.
    self.safe_act = bool(os.environ.get('DQN_SAFE_ACT'))

    if os.path.isfile(MODEL_FILE):
        self.model, payload = load_checkpoint(MODEL_FILE, self.device)
        self.arch = payload.get('arch', self.arch)
        self.logger.info(
            f"Loaded {self.arch} model ({n_parameters(self.model)} parameters) "
            f"trained for {payload.get('rounds', 0)} rounds, on {self.device}."
        )
    else:
        if not self.train:
            self.logger.warning('No model found -- the agent will act on random weights.')
        self.model = build_model(self.arch).to(self.device)
        self.logger.info(f'Starting from a fresh {self.arch} model '
                         f'({n_parameters(self.model)} parameters) on {self.device}.')

    self.model.eval()


def to_batch(self, features):
    """One ``(board, scalars)`` pair -> the batch-of-one tensors the net expects."""
    board, scalars = features
    board_t = torch.from_numpy(board).unsqueeze(0).to(self.device)
    scalar_t = torch.from_numpy(scalars).unsqueeze(0).to(self.device)
    return board_t, scalar_t


def q_values(self, features):
    """Q-row for a state, as a numpy array of length 6."""
    with torch.no_grad():
        return self.model(*to_batch(self, features)).squeeze(0).cpu().numpy()


def features_for(self, game_state):
    """Encode a state, reusing the last encoding when it is the same state.

    The framework hands the same state to ``act`` and then again to
    ``game_events_occurred`` as the *new* state of the previous step. Encoding costs
    several breadth-first searches, so a one-entry cache keyed on (round, step) halves
    the per-step feature cost during training.
    """
    if game_state is None:
        return None

    key = (game_state['round'], game_state['step'])
    cached = getattr(self, 'feature_cache', None)
    if cached is not None and cached[0] == key:
        return cached[1]

    features = state_to_features(game_state)
    self.feature_cache = (key, features)
    return features


def act(self, game_state: dict) -> str:
    """Pick a move: epsilon-greedy while training, greedy in a tournament."""
    features = features_for(self, game_state)

    epsilon = getattr(self, 'epsilon', 0.0) if self.train else 0.0
    if self.train and self.rng.random() < epsilon:
        self.logger.debug('Exploring.')
        return explore_action(self, game_state)

    q = q_values(self, features)

    if self.safe_act:
        allowed = [i for i, a in enumerate(ACTIONS) if not is_suicide(game_state, a)]
        if allowed:
            masked = np.full_like(q, -np.inf)
            masked[allowed] = q[allowed]
            q = masked

    # Break ties at random so the agent does not get stuck facing one direction.
    best = np.flatnonzero(q == q.max())
    action = ACTIONS[self.rng.choice(best)]
    self.logger.debug(f'Q={np.round(q, 2).tolist()} -> {action}')
    return action


def explore_action(self, game_state):
    """Exploration that is biased away from obviously fatal moves.

    Uniform exploration spends most of its time walking into fire or dropping
    unescapable bombs, which fills the replay buffer with deaths that all teach the
    same one lesson.  We keep the randomness but drop the moves the rules already
    prove are suicide, so exploration spends its budget on the open question --
    where to go -- rather than on rediscovering that fire is hot.
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
