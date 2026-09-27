# my_agent — tabular Q-learning

## Files

| File | Role |
|---|---|
| `callbacks.py` | `setup` / `act`, and the state → feature encoding. Loaded in every game. |
| `game_utils.py` | Exact re-implementation of the bomb/explosion rules, plus the BFS helpers. |
| `train.py` | n-step Q-learning, reward shaping, checkpointing. Loaded only with `--train`. |
| `q_table.pt` | The learned Q-table (pickled `dict`). **This is the trained model.** |
| `training_history.pt` | Per-round statistics, used to draw the training curves. |

## The model

A **tabular Q-function** over a hand-designed 9-field situation tuple. The board is
compressed so that two boards that call for the same move collapse to the same entry:

| Field | Values | Meaning |
|---|---|---|
| `target_dir` | 0–4 | direction to head (none / up / right / down / left) |
| `target_kind` | 0–3 | why: escape-or-none / coin / crate / enemy |
| `tile_here` | 0–2 | our own tile: free / blocked / on fire or about to be |
| `tile_{up,right,down,left}` | 0–2 | the same for each neighbour |
| `bomb_ok` | 0–1 | a bomb here would be legal *and* survivable |
| `bomb_worth` | 0–2 | it would hit nothing / crates / an opponent |

That is ≈29 000 reachable situations × 6 actions, small enough to fill in by
experience and large enough to express real tactics.

When we are standing in a blast radius, `target_dir` switches from "where the loot is"
to "where the nearest safe tile is", and `tile_here == 2` tells the agent which of the
two readings applies. One field therefore does duty in both the calm and the panicking
case, which keeps the table small.

## Timing model

Derived from `environment.py`, and the part that is easiest to get subtly wrong:

- A bomb observed with countdown `t` detonates **after our move `t + 1`**, and its fire
  is lethal at the end of moves `t + 1` *and* `t + 2`.
- `explosion_map[x, y] >= 1` means fire is already burning and kills us at the end of
  move 1 — but not after that.
- Dropping a bomb costs move 1; the blast lands after move 5, so we get four moves to run.

`can_survive()` does a time-indexed BFS over `(position, move number)` using exactly
these rules, which is what makes `bomb_ok` trustworthy rather than a heuristic.

## Training

```bash
python train_agent.py --fresh        # full four-stage curriculum
python train_agent.py --stages 4     # only the hardest stage, keeping the Q-table
python evaluate_agent.py             # 200 rounds vs 3 rule_based_agents, no learning
python plot_training.py              # training curves
python tests/test_features.py        # feature/timing correctness tests
```

Hyperparameters can be overridden without editing code, which is how the curriculum
retunes exploration per stage:

```bash
MY_AGENT_ALPHA=0.05 MY_AGENT_NSTEP=8 python main.py play --no-gui \
    --agents my_agent --train 1 --n-rounds 2000
```

Available: `ALPHA`, `GAMMA`, `NSTEP`, `EPS_START`, `EPS_END`, `EPS_DECAY`,
`REPLAY_SIZE`, `REPLAY_BATCH`, `SAVE_EVERY`, `QUIET`.

## Note on checkpointing

The Q-table is written every `SAVE_EVERY` rounds (default 250). There is no
end-of-training hook in the framework, so prefer round counts that are a multiple of
`SAVE_EVERY`, or the last few rounds of learning are not persisted.

## Reading the training numbers

Training-time suicide rate is **not** a measure of policy quality here, and it is worth
saying so explicitly before anyone panics at the training log.

Escaping a bomb is a committed four-move sequence. Under epsilon-greedy, the chance that
all four moves are greedy is `(1 - eps)^4` -- at `eps = 0.22` that is only 37%, so most
bomb drops during training get perturbed mid-escape and end in death. The single-step
`is_suicide` filter used during exploration cannot prevent this: every individual move
it allows really is survivable, but a later random move can still throw the escape away.

Measured on the stage-2 checkpoint (loot-crate, 100 rounds):

| | suicide rate | coins/round |
|---|---|---|
| during training (`eps` ~ 0.22) | 86.3% | 10.2 |
| greedy evaluation (`eps` = 0) | **0.0%** | **20.7** |

So always judge the agent with `evaluate_agent.py`, which runs with learning off and
`eps = 0`. Any task needing an n-step committed sequence will look far worse during
training than it actually is.
