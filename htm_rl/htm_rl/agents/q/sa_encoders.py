import numpy as np

from htm_rl.agents.q.sa_encoder import SaEncoder
from htm_rl.common.sdr import SparseSdr
from htm_rl.common.sdr_encoders import IntBucketEncoder, SdrConcatenator
from htm_rl.envs.env import Env
from htm_rl.htm_plugins.spatial_pooler import SpatialPooler


class SpSaEncoder(SaEncoder):
    state_sp: SpatialPooler
    action_encoder: IntBucketEncoder
    s_a_concatenator: SdrConcatenator
    sa_sp: SpatialPooler

    def __init__(
            self,
            env: Env,
            seed: int,
            state_sp: dict,
            action_encoder: dict,
            sa_sp: dict
    ):
        self.state_sp = SpatialPooler(input_source=env, seed=seed, **state_sp)
        action_encoder['bucket_size'] = max(
            int(.75 * self.state_sp.n_active_bits),
            action_encoder.get('bucket_size', 0)
        )
        self.action_encoder = IntBucketEncoder(n_values=env.n_actions, **action_encoder)
        self.s_a_concatenator = SdrConcatenator(input_sources=[
            self.state_sp,
            self.action_encoder
        ])
        self.sa_sp = SpatialPooler(input_source=self.s_a_concatenator, seed=seed, **sa_sp)

    def encode_state(self, state: SparseSdr, learn: bool) -> SparseSdr:
        return self.state_sp.compute(state, learn=learn)

    def encode_action(self, action: int, learn: bool) -> SparseSdr:
        return self.action_encoder.encode(action)

    def concat_s_a(self, s: SparseSdr, a: SparseSdr, learn: bool) -> SparseSdr:
        return self.s_a_concatenator.concatenate(s, a)

    def cut_s(self, s_a) -> SparseSdr:
        # assumes concatenation of (s, ...)
        # it doesn't matter what goes after state
        state_part_size = self.state_sp.output_sdr_size
        if not isinstance(s_a, np.ndarray):
            s_a = np.array(list(s_a))
        return s_a[s_a < state_part_size].copy()

    def encode_s_a(self, s_a: SparseSdr, learn: bool) -> SparseSdr:
        return self.sa_sp.compute(s_a, learn=learn)

    def concat_s_action(self, s: SparseSdr, action: int, learn: bool) -> SparseSdr:
        a = self.action_encoder.encode(action)
        s_a = self.s_a_concatenator.concatenate(s, a)
        return s_a

    def encode_s_action(self, s: SparseSdr, action: int, learn: bool) -> SparseSdr:
        s_a = self.concat_s_action(s, action, learn=learn)
        sa = self.sa_sp.compute(s_a, learn=learn)
        return sa

    @property
    def output_sdr_size(self):
        return self.sa_sp.output_sdr_size

    @property
    def s_output_sdr_size(self):
        return self.state_sp.output_sdr_size


class CrossSaEncoder(SaEncoder):
    class CrossProductEncoder:
        n_actions: int
        n_active_bits: int
        input_sdr_size: int
        output_sdr_size: int

        def __init__(self, env):
            self.n_actions = env.n_actions
            self.input_sdr_size = env.output_sdr_size
            self.output_sdr_size = env.output_sdr_size * env.n_actions
            self.n_active_bits = len(env.observe())

        def encode(self, s: SparseSdr, action: int):
            bucket_size = self.n_active_bits
            state = s[0] // bucket_size

            lft = (state * self.n_actions + action) * bucket_size
            rht = lft + bucket_size
            return np.arange(lft, rht)

        def decode_action(self, a: SparseSdr) -> int:
            return a[0] // self.n_active_bits

        def decode_s(self, sa_superposition):
            if not sa_superposition:
                return []

            stride = self.n_actions * self.n_active_bits
            state = -1
            for x in sa_superposition:
                state = x // stride
                break
            bucket_size = self.n_active_bits
            lft = state * bucket_size
            rgt = lft + bucket_size
            return np.arange(lft, rgt)

    action_encoder: IntBucketEncoder

    sa_encoder: CrossProductEncoder

    def __init__(self, env: Env):
        self.sa_encoder = self.CrossProductEncoder(env)
        self.action_encoder = IntBucketEncoder(
            n_values=env.n_actions, bucket_size=self.sa_encoder.n_active_bits
        )

    def encode_state(self, state: SparseSdr, learn: bool) -> SparseSdr:
        return state

    def encode_action(self, action: int, learn: bool) -> SparseSdr:
        return self.action_encoder.encode(action)

    def concat_s_a(self, s: SparseSdr, a: SparseSdr, learn: bool) -> SparseSdr:
        action = self.sa_encoder.decode_action(a)
        return self.sa_encoder.encode(s, action)

    def cut_s(self, sa_superposition: SparseSdr) -> SparseSdr:
        return self.sa_encoder.decode_s(sa_superposition)

    def encode_s_a(self, s_a: SparseSdr, learn: bool) -> SparseSdr:
        return s_a

    def concat_s_action(self, s: SparseSdr, action: int, learn: bool) -> SparseSdr:
        s_a = self.sa_encoder.encode(s, action)
        return s_a

    def encode_s_action(self, s: SparseSdr, action: int, learn: bool) -> SparseSdr:
        sa = s_a = self.concat_s_action(s, action, learn=learn)
        return sa

    @property
    def output_sdr_size(self):
        return self.sa_encoder.output_sdr_size

    @property
    def s_output_sdr_size(self):
        return self.sa_encoder.input_sdr_size


def make_sa_encoder(
        env: Env, seed: int, sa_encoder_config: dict
):
    if sa_encoder_config:
        return SpSaEncoder(env, seed, **sa_encoder_config)
    else:
        return CrossSaEncoder(env)
