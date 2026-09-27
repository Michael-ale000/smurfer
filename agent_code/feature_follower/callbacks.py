"""
Scripted control: an agent that does nothing but obey the hand-crafted features.

This exists to answer one question about the learning agents -- *does the network
contribute anything beyond the features we hand it?*  It is the strongest possible
"BFS-only" policy built from exactly the information `my_agent` and `dqn_agent`
receive, with no learned component whatsoever:

    in danger              -> walk the way `target_dir` points (the escape route)
    a bomb here would pay  -> drop it, but only if `bomb_ok` says it is survivable
    otherwise              -> walk the way `target_dir` points (coin / crate / enemy)

Every input it uses is one of the nine tabular features.  It has no Q-table, no
network, no parameters and no training.  If the learned agents score no better than
this, the features are doing all the work.  If they score better, the difference is
what learning bought.

Not a tournament entry: it contains no machine learning and would be rejected under
the project rules.  It is a measuring instrument.
"""

import numpy as np

from agent_code.my_agent.callbacks import state_to_features

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


def setup(self):
    self.rng = np.random.default_rng()
    self.logger.info('Scripted feature-follower: no model, no learning.')


def policy(features):
    """The whole agent. Nine features in, one action out, no parameters."""
    target_dir, kind, tile_here, up, right, down, left, bomb_ok, bomb_worth = features

    # Survival first: target_dir points at the nearest safe tile when tile_here == 2.
    if tile_here == 2:
        return ACTIONS[target_dir - 1] if target_dir != 0 else 'WAIT'

    # Bomb whenever it is both worthwhile and survivable -- the greedy reading
    # of exactly what bomb_ok and bomb_worth are telling us.
    if bomb_ok and bomb_worth > 0:
        return 'BOMB'

    # Otherwise walk toward whatever the BFS found.
    return ACTIONS[target_dir - 1] if target_dir != 0 else 'WAIT'


def act(self, game_state: dict) -> str:
    action = policy(state_to_features(game_state))
    self.logger.debug(f'-> {action}')
    return action
