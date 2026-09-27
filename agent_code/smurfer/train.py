"""
Training code: n-step Double DQN with a dueling network and uniform experience replay.

Loaded only under ``--train``.  The learning rule is the standard modern DQN stack, and
each piece is here for a reason that Bomberman makes concrete:

*Experience replay.*  Consecutive steps in one round are almost the same board, so
training on them in order feeds the optimiser a stream of highly correlated samples and
the network chases whatever it saw last.  Sampling uniformly from a large buffer breaks
that correlation and lets one hard-won episode (a kill, a clean escape) be learned from
many times.

*A target network.*  The bootstrap target contains the network's own output, so
gradient steps move the target as well as the prediction.  Freezing a copy for
``TARGET_SYNC`` updates gives the regression a stationary thing to fit.

*Double DQN.*  ``max_a Q(s', a)`` takes the maximum over six noisy estimates, which is
biased upward, and in a game where one action in six is "drop a bomb" that bias shows
up as an agent that thinks bombing is always brilliant.  Choosing the action with the
online net and evaluating it with the target net removes most of that bias.

*n-step returns.*  A bomb pays off four moves after it is dropped.  One-step updates
move that credit one tile per pass; ``N_STEP`` rewards accumulated before bootstrapping
move it in one.

*Reward shaping.*  Identical in spirit to the tabular agent's, and deliberately derived
from the same exact rule simulation, so the two agents can be compared on equal footing.
Penalties for moving away from a target are slightly larger than the reward for moving
toward one -- otherwise an agent can farm reward by oscillating between two tiles.
"""

import os
import pickle
from collections import deque, defaultdict
from typing import List

import numpy as np
import torch
import torch.nn.functional as F

import events as e
from .callbacks import ACTIONS, HISTORY_FILE, MODEL_FILE, features_for
from .features import BOARD_SHAPE, N_SCALARS, dequantise, quantise
from .game_utils import LARGE, bomb_deadlines, can_survive, crates_hit, opponents_hit
from .model import build_model, n_parameters, save_checkpoint

# --- hyperparameters ---------------------------------------------------------
# Every value can be overridden from the environment so a curriculum stage or a
# hyperparameter sweep can retune training without editing this file.


def _env(name, default, cast=float):
    return cast(os.environ.get(f'DQN_{name}', default))


N_STEP = _env('NSTEP', 3, int)            # rewards accumulated before bootstrapping
GAMMA = _env('GAMMA', 0.95)               # discount
LR = _env('LR', 2.5e-4)                   # Adam step size
BATCH = _env('BATCH', 32, int)            # transitions per gradient step (~12 ms on CPU)
TRAIN_EVERY = _env('TRAIN_EVERY', 4, int)  # environment steps between gradient steps
LEARN_START = _env('LEARN_START', 2000, int)   # fill the buffer before learning starts
TARGET_SYNC = _env('TARGET_SYNC', 1000, int)   # gradient steps between target refreshes
GRAD_CLIP = _env('GRAD_CLIP', 10.0)
REPLAY_SIZE = _env('REPLAY_SIZE', 40_000, int)  # ~5.2 kB per transition -> ~210 MB

EPS_START = _env('EPS_START', 0.90)       # exploration at round 1
EPS_END = _env('EPS_END', 0.05)           # exploration floor
EPS_DECAY_ROUNDS = _env('EPS_DECAY', 4000, int)   # rounds to decay from start to end

SAVE_EVERY = _env('SAVE_EVERY', 100, int)  # rounds between checkpoints

# Q-values are regressed directly, so the reward scale sets the scale of the outputs.
# Raw rewards here run to +/-20; shrinking them keeps the head in the range where a
# freshly initialised network already lives, which measurably speeds up early learning.
REWARD_SCALE = _env('REWARD_SCALE', 0.1)

# --- auxiliary events --------------------------------------------------------
MOVED_TOWARD_TARGET = 'MOVED_TOWARD_TARGET'
MOVED_AWAY_FROM_TARGET = 'MOVED_AWAY_FROM_TARGET'
ESCAPED_DANGER = 'ESCAPED_DANGER'
STAYED_IN_DANGER = 'STAYED_IN_DANGER'
ENTERED_DANGER = 'ENTERED_DANGER'
GOOD_BOMB_CRATE = 'GOOD_BOMB_CRATE'
GOOD_BOMB_ENEMY = 'GOOD_BOMB_ENEMY'
SUICIDAL_BOMB = 'SUICIDAL_BOMB'
POINTLESS_BOMB = 'POINTLESS_BOMB'
STUCK_IN_LOOP = 'STUCK_IN_LOOP'

GAME_REWARDS = {
    # --- real game outcomes ---
    e.COIN_COLLECTED: 3.0,
    e.KILLED_OPPONENT: 15.0,
    e.CRATE_DESTROYED: 1.0,
    e.COIN_FOUND: 0.5,
    e.SURVIVED_ROUND: 3.0,
    e.KILLED_SELF: -20.0,
    e.GOT_KILLED: -15.0,
    e.INVALID_ACTION: -2.0,
    e.WAITED: -0.2,
    # --- small step cost so dawdling is never free ---
    e.MOVED_UP: -0.05,
    e.MOVED_DOWN: -0.05,
    e.MOVED_LEFT: -0.05,
    e.MOVED_RIGHT: -0.05,
    # --- shaping ---
    MOVED_TOWARD_TARGET: 0.6,
    MOVED_AWAY_FROM_TARGET: -0.7,
    ESCAPED_DANGER: 1.5,
    STAYED_IN_DANGER: -1.0,
    ENTERED_DANGER: -1.5,
    GOOD_BOMB_CRATE: 1.5,
    GOOD_BOMB_ENEMY: 4.0,
    SUICIDAL_BOMB: -8.0,
    POINTLESS_BOMB: -1.5,
    STUCK_IN_LOOP: -1.0,
}


# -----------------------------------------------------------------------------
# Replay buffer
# -----------------------------------------------------------------------------

class ReplayBuffer:
    """A fixed-size ring buffer of transitions, stored in pre-allocated arrays.

    Boards are kept as uint8 (every plane is already in [0, 1], see features.py) and
    converted back to float only for the sampled minibatch.  A list of numpy arrays
    would cost four times the memory and fragment the heap; this way the whole buffer
    is four contiguous blocks whose size is known at startup.
    """

    def __init__(self, capacity, board_shape=BOARD_SHAPE, n_scalars=N_SCALARS):
        self.capacity = capacity
        self.boards = np.zeros((capacity, *board_shape), dtype=np.uint8)
        self.next_boards = np.zeros((capacity, *board_shape), dtype=np.uint8)
        self.scalars = np.zeros((capacity, n_scalars), dtype=np.float32)
        self.next_scalars = np.zeros((capacity, n_scalars), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.returns = np.zeros(capacity, dtype=np.float32)
        self.steps = np.zeros(capacity, dtype=np.int64)     # how many rewards are in `returns`
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.size = 0
        self.head = 0

    def __len__(self):
        return self.size

    def push(self, features, action, g, next_features, n):
        i = self.head
        board, scalars = features
        self.boards[i] = quantise(board)
        self.scalars[i] = scalars
        self.actions[i] = action
        self.returns[i] = g
        self.steps[i] = n

        if next_features is None:
            self.dones[i] = 1.0
            self.next_boards[i] = 0
            self.next_scalars[i] = 0.0
        else:
            next_board, next_scalars = next_features
            self.dones[i] = 0.0
            self.next_boards[i] = quantise(next_board)
            self.next_scalars[i] = next_scalars

        self.head = (self.head + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, rng):
        idx = rng.integers(0, self.size, size=batch_size)
        return (
            dequantise(self.boards[idx]),
            self.scalars[idx],
            self.actions[idx],
            self.returns[idx],
            dequantise(self.next_boards[idx]),
            self.next_scalars[idx],
            self.steps[idx],
            self.dones[idx],
        )


# -----------------------------------------------------------------------------
# Framework callbacks
# -----------------------------------------------------------------------------

def setup_training(self):
    """Initialise everything that only exists while learning."""
    self.model.train()

    self.target_model = build_model(self.arch).to(self.device)
    self.target_model.load_state_dict(self.model.state_dict())
    self.target_model.eval()

    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LR)
    # Each curriculum stage is a separate process. Restoring Adam's moment estimates
    # means a later stage continues the optimisation instead of restarting it with a
    # cold, effectively much larger, first step.
    if os.path.isfile(MODEL_FILE):
        payload = torch.load(MODEL_FILE, map_location=self.device, weights_only=False)
        if payload.get('optimizer') is not None and payload.get('arch') == self.arch:
            try:
                self.optimizer.load_state_dict(payload['optimizer'])
                self.logger.info('Restored optimiser state from the checkpoint.')
            except ValueError:
                self.logger.warning('Optimiser state did not fit the model; starting fresh.')

    self.replay = ReplayBuffer(REPLAY_SIZE)
    self.n_step_buffer = deque(maxlen=N_STEP)
    self.coordinate_history = deque(maxlen=10)

    self.epsilon = EPS_START
    self.rounds_done = 0
    self.env_steps = 0
    self.grad_steps = 0

    # Per-round bookkeeping, appended to self.history for the training curve. We
    # continue the existing history rather than starting a new one -- otherwise every
    # curriculum stage would overwrite the curve drawn by the stage before it.
    self.history = load_history()
    self.rounds_before = len(self.history)
    self.round_stats = defaultdict(float)

    self.logger.info(
        f'Training initialised: {self.arch} net, {n_parameters(self.model)} parameters, '
        f'n-step={N_STEP}, batch={BATCH}, lr={LR}, buffer={REPLAY_SIZE}, device={self.device}.'
    )


def game_events_occurred(self, old_game_state: dict, self_action: str,
                         new_game_state: dict, events: List[str]):
    """Turn one environment step into stored experience and (sometimes) a gradient step."""
    if old_game_state is None:
        return

    old_features = features_for(self, old_game_state)
    new_features = features_for(self, new_game_state)

    events = events + auxiliary_events(self, old_game_state, old_features, self_action,
                                       new_game_state, events)
    reward = reward_from_events(self, events)

    record_stats(self, events, reward)
    push_transition(self, old_features, self_action, reward, new_features)

    self.env_steps += 1
    if self.env_steps % TRAIN_EVERY == 0:
        learn(self)


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """Final transition of the episode, then flush, learn and checkpoint."""
    last_features = features_for(self, last_game_state)
    events = events + auxiliary_events(self, last_game_state, last_features, last_action,
                                       None, events)
    reward = reward_from_events(self, events)

    record_stats(self, events, reward)
    push_transition(self, last_features, last_action, reward, None)

    # Drain the n-step buffer: the remaining tail has no bootstrap value.
    while self.n_step_buffer:
        store_n_step(self)
        self.n_step_buffer.popleft()

    learn(self)

    self.rounds_done += 1
    self.epsilon = max(
        EPS_END,
        EPS_START - (EPS_START - EPS_END) * self.rounds_done / EPS_DECAY_ROUNDS,
    )

    log_round(self, last_game_state)
    self.coordinate_history.clear()
    self.n_step_buffer.clear()

    if self.rounds_done % SAVE_EVERY == 0:
        save_model(self)


# -----------------------------------------------------------------------------
# Experience
# -----------------------------------------------------------------------------

def push_transition(self, features, action, reward, next_features):
    """Add a step and emit an n-step transition once enough rewards have accumulated."""
    if features is None or action is None:
        return

    self.n_step_buffer.append((features, ACTIONS.index(action),
                               reward * REWARD_SCALE, next_features))
    if len(self.n_step_buffer) == N_STEP:
        store_n_step(self)


def store_n_step(self):
    """Write the oldest entry of the n-step buffer into replay as one transition."""
    features, action, _, _ = self.n_step_buffer[0]

    g = 0.0
    for k, (_, _, reward, _) in enumerate(self.n_step_buffer):
        g += (GAMMA ** k) * reward

    _, _, _, tail_features = self.n_step_buffer[-1]
    self.replay.push(features, action, g, tail_features, len(self.n_step_buffer))


# -----------------------------------------------------------------------------
# Learning
# -----------------------------------------------------------------------------

def learn(self):
    """One gradient step of n-step Double DQN on a uniformly sampled minibatch."""
    if len(self.replay) < max(LEARN_START, BATCH):
        return

    boards, scalars, actions, returns, next_boards, next_scalars, steps, dones = \
        self.replay.sample(BATCH, self.rng)

    device = self.device
    boards = torch.from_numpy(boards).to(device)
    scalars = torch.from_numpy(scalars).to(device)
    actions = torch.from_numpy(actions).to(device)
    returns = torch.from_numpy(returns).to(device)
    next_boards = torch.from_numpy(next_boards).to(device)
    next_scalars = torch.from_numpy(next_scalars).to(device)
    discounts = torch.from_numpy((GAMMA ** steps).astype(np.float32)).to(device)
    dones = torch.from_numpy(dones).to(device)

    q = self.model(boards, scalars).gather(1, actions.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        # Double DQN: the online net picks the action, the frozen net scores it.
        best = self.model(next_boards, next_scalars).argmax(dim=1, keepdim=True)
        next_q = self.target_model(next_boards, next_scalars).gather(1, best).squeeze(1)
        target = returns + (1.0 - dones) * discounts * next_q

    # Huber rather than MSE: a single mispredicted death is a huge residual, and
    # squaring it would let one transition dominate the whole minibatch.
    loss = F.smooth_l1_loss(q, target)

    self.optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(self.model.parameters(), GRAD_CLIP)
    self.optimizer.step()

    self.grad_steps += 1
    self.round_stats['loss_sum'] += float(loss.item())
    self.round_stats['loss_n'] += 1
    self.round_stats['q_sum'] += float(q.mean().item())

    if self.grad_steps % TARGET_SYNC == 0:
        self.target_model.load_state_dict(self.model.state_dict())
        self.logger.debug(f'Target network synced at {self.grad_steps} gradient steps.')


# -----------------------------------------------------------------------------
# Reward shaping
# -----------------------------------------------------------------------------

def auxiliary_events(self, old_game_state, old_features, self_action, new_game_state, events):
    """Derive shaping events from the same information the features expose."""
    extra = []
    if old_game_state is None or old_features is None:
        return extra

    x, y = old_game_state['self'][3]
    old_deadlines = bomb_deadlines(old_game_state)
    was_in_danger = old_deadlines[x, y] < LARGE

    # --- bomb quality ---------------------------------------------------------
    if e.BOMB_DROPPED in events:
        with_bomb = bomb_deadlines(old_game_state, extra_bomb=(x, y))
        if not can_survive(old_game_state, (x, y), with_bomb, extra_wait_first=True):
            extra.append(SUICIDAL_BOMB)
        elif opponents_hit(old_game_state, x, y) > 0:
            extra.append(GOOD_BOMB_ENEMY)
        elif crates_hit(old_game_state['field'], x, y) > 0:
            extra.append(GOOD_BOMB_CRATE)
        else:
            extra.append(POINTLESS_BOMB)

    # --- danger transitions ---------------------------------------------------
    if new_game_state is not None:
        nx, ny = new_game_state['self'][3]
        now_in_danger = bomb_deadlines(new_game_state)[nx, ny] < LARGE

        if was_in_danger and not now_in_danger:
            extra.append(ESCAPED_DANGER)
        elif was_in_danger and now_in_danger:
            extra.append(STAYED_IN_DANGER)
        elif not was_in_danger and now_in_danger and e.BOMB_DROPPED not in events:
            # Dropping your own bomb necessarily puts you in danger; don't punish that.
            extra.append(ENTERED_DANGER)

        # --- progress toward the current objective ---------------------------
        # scalars[0:5] is the one-hot BFS direction: index 0 is "no target".
        _, old_scalars = old_features
        target_dir = int(np.argmax(old_scalars[:5]))
        if target_dir != 0 and self_action in ACTIONS[:4]:
            moved = ACTIONS.index(self_action) + 1
            if (nx, ny) != (x, y):
                extra.append(MOVED_TOWARD_TARGET if moved == target_dir
                             else MOVED_AWAY_FROM_TARGET)

        # --- loop detection ---------------------------------------------------
        self.coordinate_history.append((nx, ny))
        if not now_in_danger and self.coordinate_history.count((nx, ny)) > 3:
            extra.append(STUCK_IN_LOOP)

    return extra


def reward_from_events(self, events: List[str]) -> float:
    """Sum the reward table over everything that happened this step."""
    total = sum(GAME_REWARDS.get(event, 0.0) for event in events)
    self.logger.debug(f'Reward {total:+.2f} for {", ".join(events)}')
    return total


# -----------------------------------------------------------------------------
# Bookkeeping
# -----------------------------------------------------------------------------

def record_stats(self, events, reward):
    self.round_stats['reward'] += reward
    self.round_stats['coins'] += events.count(e.COIN_COLLECTED)
    self.round_stats['kills'] += events.count(e.KILLED_OPPONENT)
    self.round_stats['crates'] += events.count(e.CRATE_DESTROYED)
    self.round_stats['invalid'] += events.count(e.INVALID_ACTION)
    if e.KILLED_SELF in events:
        self.round_stats['suicide'] = 1
    if e.SURVIVED_ROUND in events:
        self.round_stats['survived'] = 1


def log_round(self, last_game_state):
    stats = dict(self.round_stats)
    loss_n = max(1.0, stats.pop('loss_n', 0.0))
    stats['loss'] = stats.pop('loss_sum', 0.0) / loss_n
    stats['q_mean'] = stats.pop('q_sum', 0.0) / loss_n
    stats['round'] = self.rounds_before + self.rounds_done
    stats['stage_round'] = self.rounds_done
    stats['steps'] = last_game_state['step'] if last_game_state else 0
    stats['score'] = last_game_state['self'][1] if last_game_state else 0
    stats['epsilon'] = self.epsilon
    stats['buffer'] = len(self.replay)
    stats['grad_steps'] = self.grad_steps
    self.history.append(stats)

    self.logger.info(
        f"Round {self.rounds_done}: score={stats['score']} "
        f"reward={stats.get('reward', 0):.1f} coins={stats.get('coins', 0):.0f} "
        f"kills={stats.get('kills', 0):.0f} crates={stats.get('crates', 0):.0f} "
        f"suicide={stats.get('suicide', 0):.0f} eps={self.epsilon:.3f} "
        f"loss={stats['loss']:.4f} Q={stats['q_mean']:.2f} "
        f"buffer={stats['buffer']} updates={self.grad_steps}"
    )
    self.round_stats = defaultdict(float)


def load_history():
    """Read the training curve written by earlier curriculum stages, if any."""
    if not os.path.isfile(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, 'rb') as f:
            return pickle.load(f)
    except (EOFError, pickle.UnpicklingError):
        return []


def save_model(self):
    save_checkpoint(MODEL_FILE, self.model, self.arch, extra={
        'optimizer': self.optimizer.state_dict(),
        'rounds': self.rounds_before + self.rounds_done,
        'grad_steps': self.grad_steps,
        'epsilon': self.epsilon,
    })

    with open(HISTORY_FILE, 'wb') as f:
        pickle.dump(self.history, f)

    self.logger.info(f'Saved model after {self.rounds_done} rounds '
                     f'({self.grad_steps} gradient steps).')
