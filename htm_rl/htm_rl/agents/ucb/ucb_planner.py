from collections import deque
from typing import List, Mapping, Set, Tuple, Deque, Dict

from htm_rl.agent.memory import Memory
from htm_rl.common.base_sa import Sa
from htm_rl.common.int_sdr_encoder import IntSdrEncoder
from htm_rl.common.sa_sdr_encoder import SaSdrEncoder
from htm_rl.common.sdr import SparseSdr
from htm_rl.common.utils import range_reverse, trace


# noinspection PyPep8Naming
from htm_rl.htm_plugins.spatial_pooler import SpatialPooler


class UcbPlanner:
    encoder: SaSdrEncoder
    spatial_pooler: SpatialPooler
    planning_horizon: int
    encoder: SaSdrEncoder

    _n_actions: int

    def __init__(
        self, encoder: SaSdrEncoder, spatial_pooler: SpatialPooler,
        n_actions: int, planning_horizon: int
    ):
        self.encoder = encoder
        self.spatial_pooler = spatial_pooler
        self.planning_horizon = planning_horizon
        self._n_actions = n_actions

    def encode_sa(self, sa: Sa, learn: bool) -> SparseSdr:
        sa_sdr = self.encoder.encode(sa)
        sa_sdr = self.spatial_pooler.encode(sa_sdr, learn=learn)
        return sa_sdr

    def predict_states(self, initial_sa: Sa, verbosity: int):
        action_outcomes = []
        if self.planning_horizon == 0:
            return action_outcomes

        trace(verbosity, 2, '\n======> Planning')

        state = initial_sa.state
        for action in range(self._n_actions):
            sa = Sa(state, action)
            sa_sdr = self.encode_sa(sa, learn=False)
            action_outcomes.append(sa_sdr)

        trace(verbosity, 2, '<====== Planning complete')
        return action_outcomes
