from typing import Dict, List

import numpy as np

from htm_rl.agents.q.sa_encoder import SaEncoder
from htm_rl.common.sdr import SparseSdr
from htm_rl.common.sdr_encoders import IntBucketEncoder, SdrConcatenator
from htm_rl.envs.env import Env
from htm_rl.htm_plugins.spatial_pooler import SpatialPooler


class SpSaEncoder(SaEncoder):
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

    def decode_state(self, sdr) -> SparseSdr:
        # assumes concatenation of (state, ...)
        # it doesn't matter what goes after state
        state_part_size = self.state_sp.output_sdr_size
        if not isinstance(sdr, np.ndarray):
            sdr = np.array(list(sdr))
        return sdr[sdr < state_part_size].copy()

    @property
    def output_sdr_size(self):
        return self.sa_sp.output_sdr_size


class CrossSaEncoder(SaEncoder):
    class CrossProductEncoder:
        n_actions: int
        n_active_bits: int
        output_sdr_size: int

        def __init__(self, env):
            self.n_actions = env.n_actions
            self.output_sdr_size = env.output_sdr_size * env.n_actions
            self.n_active_bits = len(env.observe())

        def encode(self, s, action):
            bucket_size = self.n_active_bits
            state = s[0] // bucket_size

            l = (state * self.n_actions + action) * bucket_size
            r = l + bucket_size
            return np.arange(l, r)

        def decode_state(self, sa_superposition):
            if not sa_superposition:
                return []

            stride = self.n_actions * self.n_active_bits
            state = -1
            for x in sa_superposition:
                state = x // stride
                break
            bucket_size = self.n_active_bits
            l = state * bucket_size
            r = l + bucket_size
            return np.arange(l, r)

    n_actions: int
    sa_encoder: CrossProductEncoder

    def __init__(self, env: Env):
        self.n_actions = env.n_actions
        self.sa_encoder = self.CrossProductEncoder(env)

    def encode_state(self, state: SparseSdr, learn: bool) -> SparseSdr:
        return state

    def encode_actions(self, s: SparseSdr, learn: bool) -> List[SparseSdr]:
        return [
            self.encode_sa(s, action, learn=learn)
            for action in range(self.n_actions)
        ]

    def encode_sa(self, s: SparseSdr, action: int, learn: bool) -> SparseSdr:
        return self.sa_encoder.encode(s, action)

    def decode_state(self, sdr) -> SparseSdr:
        return self.sa_encoder.decode_state(sdr)

    @property
    def output_sdr_size(self):
        return self.sa_encoder.output_sdr_size
