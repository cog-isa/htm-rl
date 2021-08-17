from typing import Dict, List

import numpy as np
from htm_rl.common.sdr import SparseSdr
from htm_rl.common.sdr_encoders import IntBucketEncoder, SdrConcatenator
from htm_rl.envs.env import Env
from htm_rl.htm_plugins.spatial_pooler import SpatialPooler


class SaEncoder:
    n_actions: int
    state_sp: SpatialPooler
    action_encoder: IntBucketEncoder
    sa_concatenator: SdrConcatenator
    sa_sp: SpatialPooler

    def __init__(
            self,
            env: Env,
            seed: int,
            state_sp: Dict,
            action_encoder: Dict,
            sa_sp: Dict
    ):
        self.state_sp = SpatialPooler(input_source=env, seed=seed, **state_sp)
        action_encoder['bucket_size'] = max(
            int(.75 * self.state_sp.n_active_bits),
            action_encoder.get('bucket_size', 0)
        )
        self.action_encoder = IntBucketEncoder(n_values=env.n_actions, **action_encoder)
        self.sa_concatenator = SdrConcatenator(input_sources=[
            self.state_sp,
            self.action_encoder
        ])
        self.sa_sp = SpatialPooler(input_source=self.sa_concatenator, seed=seed, **sa_sp)
        self.n_actions = env.n_actions

    def encode_state(self, state: SparseSdr, learn: bool) -> SparseSdr:
        return self.state_sp.compute(state, learn=learn)

    def encode_actions(self, s: SparseSdr, learn: bool) -> List[SparseSdr]:
        actions = []
        for action in range(self.n_actions):
            sa_sdr = self.encode_sa(s, action, learn=learn)
            actions.append(sa_sdr)

        return actions

    def encode_sa(self, s: SparseSdr, action: int, learn: bool) -> SparseSdr:
        a = self.action_encoder.encode(action)
        s_a = self.sa_concatenator.concatenate(s, a)
        sa = self.sa_sp.compute(s_a, learn=learn)
        return sa

    def split_s_sa(self, s_a: SparseSdr) -> SparseSdr:
        state_part_size = self.state_sp.output_sdr_size
        if not isinstance(s_a, np.ndarray):
            s_a = np.array(list(s_a))
        return s_a[s_a < state_part_size].copy()

    @property
    def output_sdr_size(self):
        return self.sa_sp.output_sdr_size
