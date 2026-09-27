"""
Training code: n-step tabular Q-learning with shaped rewards.

Two things make the sparse Bomberman reward learnable here.

*n-step returns.*  A single-step update propagates credit one tile per episode, so a
bomb that pays off four moves later takes many rounds to be reinforced.  Accumulating
``N_STEP`` rewards before bootstrapping moves that credit back in one update.

*Reward shaping.*  Coins and kills alone are far too sparse.  We add auxiliary events
derived from the same features the agent sees, so the shaping never tells it anything
its state representation cannot represent.  Penalties for moving away from a target are
set slightly larger than the reward for moving toward one -- otherwise an agent can farm
reward by oscillating between two tiles instead of making progress.
"""

import os
import pickle
from collections import deque, defaultdict
from typing import List

import numpy as np

import events as e
from .callbacks import ACTIONS, MODEL_FILE, q_values, state_to_features
from .game_utils import LARGE, bomb_deadlines, can_survive, crates_hit, opponents_hit

# --- hyperparameters ---------------------------------------------------------
# Every value can be overridden from the environment so a curriculum stage or a
# hyperparameter sweep can retune training without editing this file.


def _env(name, default, cast=float):
    return cast(os.environ.get(f'MY_AGENT_{name}', default))


N_STEP = _env('NSTEP', 5, int)          # how far returns accumulate before bootstrapping
GAMMA = _env('GAMMA', 0.95)             # discount
ALPHA = _env('ALPHA', 0.1)              # learning rate
EPS_START = _env('EPS_START', 0.85)     # exploration at round 1
EPS_END = _env('EPS_END', 0.05)         # exploration floor
EPS_DECAY_ROUNDS = _env('EPS_DECAY', 8000, int)  # rounds to decay from start to end

REPLAY_SIZE = _env('REPLAY_SIZE', 50_000, int)   # transitions retained for sweeps
REPLAY_BATCH = _env('REPLAY_BATCH', 512, int)    # transitions replayed per round
SAVE_EVERY = _env('SAVE_EVERY', 250, int)        # rounds between checkpoints

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


def setup_training(self):
    """Initialise everything that only exists while learning."""
    self.n_step_buffer = deque(maxlen=N_STEP)
    self.replay = deque(maxlen=REPLAY_SIZE)
    self.coordinate_history = deque(maxlen=10)

    self.epsilon = EPS_START
    self.rounds_done = 0

    # Per-round bookkeeping, appended to self.history for the training curve.
    # Each curriculum stage is a separate process, so we continue the existing
    # history rather than starting a new one -- otherwise every stage would
    # overwrite the curve drawn by the stage before it.
    self.history = load_history()
    self.rounds_before = len(self.history)
    self.round_stats = defaultdict(float)

    self.logger.info('Training initialised.')


def game_events_occurred(self, old_game_state: dict, self_action: str,
                         new_game_state: dict, events: List[str]):
    """Turn one environment step into a learning update."""
    if old_game_state is None:
        return

    old_features = state_to_features(old_game_state)
    new_features = state_to_features(new_game_state)

    events = events + auxiliary_events(self, old_game_state, self_action, new_game_state, events)
    reward = reward_from_events(self, events)

    record_stats(self, events, reward)
    push_transition(self, old_features, self_action, reward, new_features)


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """Final update of the episode, then flush, replay and checkpoint."""
    last_features = state_to_features(last_game_state)
    events = events + auxiliary_events(self, last_game_state, last_action, None, events)
    reward = reward_from_events(self, events)

    record_stats(self, events, reward)
    push_transition(self, last_features, last_action, reward, None)

    # Drain the n-step buffer: the remaining tail has no bootstrap value.
    while self.n_step_buffer:
        apply_update(self, *n_step_return(self))
        self.n_step_buffer.popleft()

    replay_sweep(self)

    self.rounds_done += 1
    self.epsilon = max(
        EPS_END,
        EPS_START - (EPS_START - EPS_END) * self.rounds_done / EPS_DECAY_ROUNDS,
    )

    log_round(self, last_game_state)
    self.coordinate_history.clear()

    if self.rounds_done % SAVE_EVERY == 0:
        save_model(self)


# -----------------------------------------------------------------------------
# Q-learning
# -----------------------------------------------------------------------------

def push_transition(self, features, action, reward, next_features):
    """Add a transition and fire an update once N_STEP rewards have accumulated."""
    if features is None or action is None:
        return

    self.n_step_buffer.append((features, action, reward, next_features))
    self.replay.append((features, action, reward, next_features))

    if len(self.n_step_buffer) == N_STEP:
        apply_update(self, *n_step_return(self))


def n_step_return(self):
    """Discounted return over the buffer plus the state it bootstraps from."""
    features, action, _, _ = self.n_step_buffer[0]

    g = 0.0
    for k, (_, _, reward, _) in enumerate(self.n_step_buffer):
        g += (GAMMA ** k) * reward

    _, _, _, tail_features = self.n_step_buffer[-1]
    return features, action, g, tail_features, len(self.n_step_buffer)


def apply_update(self, features, action, g, next_features, n):
    """Tabular Q-learning update with an n-step target."""
    if next_features is not None:
        g += (GAMMA ** n) * q_values(self, next_features).max()

    q = q_values(self, features)
    idx = ACTIONS.index(action)
    q[idx] += ALPHA * (g - q[idx])


def replay_sweep(self):
    """Extra one-step updates on remembered transitions to squeeze more out of them."""
    if len(self.replay) < REPLAY_BATCH:
        return

    idx = np.random.randint(0, len(self.replay), size=REPLAY_BATCH)
    for i in idx:
        features, action, reward, next_features = self.replay[i]
        apply_update(self, features, action, reward, next_features, 1)


# -----------------------------------------------------------------------------
# Reward shaping
# -----------------------------------------------------------------------------

def auxiliary_events(self, old_game_state, self_action, new_game_state, events):
    """Derive shaping events from the same information the features expose."""
    extra = []
    if old_game_state is None:
        return extra

    old_features = state_to_features(old_game_state)
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
        target_dir = old_features[0]
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
    stats['round'] = self.rounds_before + self.rounds_done
    stats['stage_round'] = self.rounds_done
    stats['steps'] = last_game_state['step'] if last_game_state else 0
    stats['score'] = last_game_state['self'][1] if last_game_state else 0
    stats['epsilon'] = self.epsilon
    stats['states'] = len(self.q_table)
    self.history.append(stats)

    self.logger.info(
        f"Round {self.rounds_done}: score={stats['score']} "
        f"reward={stats.get('reward', 0):.1f} coins={stats.get('coins', 0):.0f} "
        f"kills={stats.get('kills', 0):.0f} crates={stats.get('crates', 0):.0f} "
        f"suicide={stats.get('suicide', 0):.0f} eps={self.epsilon:.3f} "
        f"|Q|={len(self.q_table)}"
    )
    self.round_stats = defaultdict(float)


def load_history():
    """Read the training curve written by earlier curriculum stages, if any."""
    history_file = MODEL_FILE.replace('q_table.pt', 'training_history.pt')
    if not os.path.isfile(history_file):
        return []
    try:
        with open(history_file, 'rb') as f:
            return pickle.load(f)
    except (EOFError, pickle.UnpicklingError):
        return []


def save_model(self):
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(self.q_table, f)

    history_file = MODEL_FILE.replace('q_table.pt', 'training_history.pt')
    with open(history_file, 'wb') as f:
        pickle.dump(self.history, f)

    self.logger.info(f'Saved Q-table ({len(self.q_table)} situations) after '
                     f'{self.rounds_done} rounds.')
