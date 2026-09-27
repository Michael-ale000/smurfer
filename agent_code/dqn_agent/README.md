# dqn_agent — deep Q-network

The second agent for the project: the same game, the same reward shaping and the same
rule simulation as `my_agent`, but the Q-function is a small convolutional network
instead of a lookup table.

## Files

| File | Role |
|---|---|
| `features.py` | state → `(board, scalars)` encoding. |
| `model.py` | the networks (`conv`, `mlp`) and the checkpoint format. |
| `callbacks.py` | `setup` / `act`, exploration. Loaded in every game. |
| `game_utils.py` | exact re-implementation of the bomb/explosion rules, plus BFS helpers. |
| `train.py` | n-step Double DQN: replay buffer, target network, shaping, checkpoints. |
| `dqn_model.pt` | the trained network (+ optimiser state). **This is the model.** |
| `training_history.pt` | per-round statistics, used to draw the training curves. |

`game_utils.py` is a copy of the one in `my_agent`, not an import: the framework loads
each agent as its own package and a tournament submission has to stand alone.

## Why a network at all

The tabular agent works, but only because its 9-field tuple throws almost everything
away. It cannot represent "there is a crate two tiles away with a corner to hide
behind", because that fact has nowhere to live in the tuple. A table forces you to
choose the abstraction up front; a network lets the abstraction be learned.

So here the state stays **spatial**:

```
board    (9, 17, 17)   wall, crate, coin, self, other, bomb, explosion, danger, walkable
scalars  (21,)         BFS target one-hot, target kind, neighbour safety, bomb facts, step, ...
```

Two planes are worth singling out. `bomb` holds each bomb's *urgency* rather than a
flag, and `danger` holds, per tile, **how soon** it turns lethal — computed by the same
time-indexed BFS (`game_utils.can_survive` / `bomb_deadlines`) the tabular agent uses.
Fire timing is the one thing in this game that is exactly knowable, and making the
network infer it from reward alone would waste most of the training budget.

The 21 scalars are the concession to that same argument: they carry the BFS target
direction and the "is a bomb here survivable" verdict, which are cheap to compute
exactly and expensive to learn. Everything else — where to go, when to bomb, when to
run — is left to the network.

All values are in `[0, 1]`, which is what lets the replay buffer store boards as
`uint8` (a factor of four less memory).

## Network

`conv` (the real agent, 454k parameters — 90% of them in the 1600→256 layer):

```
board (9,17,17) ─ conv3x3 16 ─ conv3x3 s2 32 ─ conv3x3 s2 64 ─ flatten(1600) ┐
                                                                              ├─ 256 ─ dueling head ─ Q(s,·) ∈ R^6
scalars (21) ──── 64 ─────────────────────────────────────────────────────────┘
```

Convolution is the right prior: "in a blast lane", "crate beside a corner" mean the
same thing wherever they appear on the map, so those weights should be shared across
positions. The head is **dueling** — `Q(s,a) = V(s) + A(s,a) - mean_a A(s,a)` — because
most Bomberman states are dominated by "how dangerous is it here", which is a property
of the state, not of the action, and should be learned once rather than six times.

`mlp` is the ablation: identical inputs, flattened, no weight sharing. It has 1.48M
parameters (three times as many) and is the natural control for "is the convolution
actually doing anything?". Run it with `--arch mlp`.

## Learning rule

n-step **Double** DQN with a target network and uniform experience replay.

| Piece | Why it is here |
|---|---|
| Experience replay (40k) | Consecutive steps are nearly identical boards; sampling uniformly breaks that correlation and lets one good escape be learned from many times. |
| Target network (sync 1000) | The bootstrap target contains the network's own output; freezing a copy gives the regression something stationary to fit. |
| Double DQN | `max_a Q` over six noisy estimates is biased upward, and with one action in six being "drop a bomb" that bias shows up directly as an agent that thinks bombing is always brilliant. |
| n-step returns (n=3) | A bomb pays off four moves after it is dropped. |
| Huber loss | One mispredicted death is a huge residual; squaring it would let a single transition dominate the batch. |
| Reward scale 0.1 | Raw rewards run to ±20. Shrinking them keeps the targets in the range a freshly initialised head already outputs. |
| Filtered exploration | Random moves are drawn only from moves the rules prove are survivable, so the exploration budget goes on the open question (where to go) rather than on rediscovering that fire is hot. |

Rewards and shaping events are **identical** to `my_agent`'s, on purpose: it is the
only way the tabular/deep comparison in the report is a comparison of *models* rather
than of reward functions.

## Training

```bash
python3 train_dqn.py --fresh              # full four-stage curriculum
python3 train_dqn.py --scale 0.25         # a quarter of the rounds, for a first look
python3 train_dqn.py --stages 4           # hardest stage only, keeping the network
python3 train_dqn.py --arch mlp --fresh   # the no-convolution ablation
python3 evaluate_agent.py --agent dqn_agent          # 200 rounds vs 3 rule_based, no learning
python3 plot_training.py --agent dqn_agent           # training curves
python3 tests/test_dqn.py                            # encoding / model / buffer tests
```

The curriculum is the same four stages as the tabular agent (coin-heaven → loot-crate →
weak opponents → `rule_based_agent`), carrying both the weights **and Adam's moment
estimates** across stages, so a later stage continues the optimisation instead of
restarting it with an effectively enormous first step. The replay buffer does not
survive a stage boundary — deliberately, since stage 3's solo transitions are
off-distribution for stage 4.

Hyperparameters are read from the environment, so a stage or a sweep can retune
training without editing code:

```bash
DQN_LR=1e-4 DQN_NSTEP=5 DQN_BATCH=64 python3 main.py play --no-gui \
    --agents dqn_agent --train 1 --n-rounds 2000
```

Available: `ARCH`, `LR`, `GAMMA`, `NSTEP`, `BATCH`, `TRAIN_EVERY`, `LEARN_START`,
`TARGET_SYNC`, `GRAD_CLIP`, `REPLAY_SIZE`, `EPS_START`, `EPS_END`, `EPS_DECAY`,
`SAVE_EVERY`, `REWARD_SCALE`, `CONV_CHANNELS`, `HIDDEN`, `DEVICE`, `THREADS`,
`SAFE_ACT`, `QUIET`.

### How long it takes

Measured on this machine (CPU, batch 32): **~12 ms per gradient step**, one gradient
step per 4 environment steps, so a full-length 400-step round costs roughly 1–2 s
including the environment itself. Early rounds are much shorter because the agent dies
early. Budget a few hours for the whole curriculum, and use `--scale` to shorten it.

`DQN_DEVICE=mps` (or `cuda`) is available but **CPU is usually faster here** — the
network is small enough that per-kernel launch overhead outweighs the arithmetic. The
default is CPU; the override exists so the claim can be measured rather than believed.

Inference costs ~0.25 ms per step, about 2000× inside the 0.5 s tournament budget.

## Note on checkpointing

The model is written every `SAVE_EVERY` rounds (default 100). There is no
end-of-training hook in the framework, so prefer round counts that are a multiple of
`SAVE_EVERY`, or the last few rounds of learning are not persisted.

## Reading the training numbers

Everything the `my_agent` README says about this applies here and is, if anything,
stronger: escaping a bomb is a committed four-move sequence, so at `eps = 0.22` only
`(1 - eps)^4 ≈ 37%` of escapes survive exploration intact. Training-time suicide rate
therefore measures the exploration schedule, not the policy. **Judge the agent with
`evaluate_agent.py`**, which runs with learning off and `eps = 0`.

Two extra numbers appear in this agent's log line and history: `loss` (mean Huber loss
over the round) and `Q` (mean predicted value of the actions actually taken). Loss
plateauing is not a sign of success on its own — the targets move — but `Q` drifting
steadily upward while the score does not is the classic signature of over-estimation,
and the first thing to check if training goes wrong.

## Known trade-offs

- **No prioritised replay.** Uniform sampling is simpler and the shaped reward already
  makes most transitions informative. PER is the obvious next thing to try.
- **No data augmentation.** The board has an eight-fold symmetry that could multiply
  the effective sample count, but transforming the board also permutes the action
  space and the direction scalars, and getting that subtly wrong silently poisons
  training. Left out rather than left wrong.
- **`DQN_SAFE_ACT=1`** masks provably-fatal moves at inference. It is off by default so
  that an evaluation measures the learned policy rather than the filter; turn it on if
  the goal is purely to win the tournament, and say so in the report if you do.
