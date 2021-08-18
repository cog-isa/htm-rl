from typing import Dict, List

import numpy as np
from common.sdr_encoders import IdEncoder
from htm_rl.common.sdr import SparseSdr
from htm_rl.common.sdr_encoders import IntBucketEncoder, SdrConcatenator
from htm_rl.envs.env import Env
from htm_rl.htm_plugins.spatial_pooler import SpatialPooler


# Forward declaration
class CrossProductEncoder: pass


class NaiveSaEncoder:
    n_actions: int
    state_sp: IdEncoder
    sa_concatenator: CrossProductEncoder
    sa_sp: IdEncoder

    def __init__(self, env: Env):
        self.state_sp = IdEncoder(input_source=env)
        self.sa_concatenator = CrossProductEncoder(
            self.state_sp, env.n_actions
        )
        self.sa_sp = IdEncoder(input_source=self.sa_concatenator)
        self.n_actions = env.n_actions

    def encode_state(self, state: SparseSdr, learn: bool) -> SparseSdr:
        return self.state_sp.encode(state)

    def encode_actions(self, s: SparseSdr, learn: bool) -> List[SparseSdr]:
        return [
            self.encode_sa(s, action, learn=learn)
            for action in range(self.n_actions)
        ]

    def encode_sa(self, s: SparseSdr, action: int, learn: bool) -> SparseSdr:
        state = self.state_sp.decode(s)
        s_a = self.sa_concatenator.encode(state, action, len(s))
        return s_a

    def split_s_sa(self, s_a: SparseSdr) -> SparseSdr:
        state_part_size = self.state_sp.output_sdr_size
        return s_a[s_a < state_part_size].copy()

    @property
    def output_sdr_size(self):
        return self.sa_sp.output_sdr_size


class CrossProductEncoder:
    output_sdr_size: int
    n_active_bits: int

    _n_actions: int

    def __init__(self, s_input_source, n_actions):
        self._n_actions = n_actions
        self.output_sdr_size = s_input_source.output_sdr_size * n_actions
        self.n_active_bits = 1

    def encode(self, state, action, bucket_size):
        self.n_active_bits = bucket_size
        l = (state * self._n_actions + action) * bucket_size
        r = l + bucket_size
        return np.arange(l, r)
