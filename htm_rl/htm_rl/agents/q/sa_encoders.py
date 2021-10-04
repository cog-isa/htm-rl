from typing import Optional

import numpy as np

from htm_rl.agents.q.sa_encoder import SaEncoder
from htm_rl.common.sdr import SparseSdr, sparse_to_dense
from htm_rl.common.sdr_encoders import IntBucketEncoder, SdrConcatenator
from htm_rl.envs.env import Env
from htm_rl.htm_plugins.spatial_pooler import SpatialPooler
from htm_rl.modules.empowerment import Memory


class SpSaEncoder(SaEncoder):
    state_sp: SpatialPooler
    action_encoder: IntBucketEncoder
    s_a_concatenator: SdrConcatenator
    sa_sp: SpatialPooler

    state_clusters: Optional[Memory]
    state_clusters_similarity_threshold_max: Optional[float]

    def __init__(
            self,
            env: Env,
            seed: int,
            state_sp: dict,
            action_encoder: dict,
            sa_sp: dict,
            state_clusters_similarity_threshold: float = None,
            state_clusters_similarity_threshold_max: float = None,
    ):
        self.state_sp = SpatialPooler(input_source=env, seed=seed, **state_sp)
        self.state_clusters = None
        if state_clusters_similarity_threshold is not None:
            self.state_clusters = Memory(
                size=self.state_sp.output_sdr_size,
                threshold=state_clusters_similarity_threshold
            )
            self.state_clusters_similarity_threshold_max = state_clusters_similarity_threshold_max
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
        s = self.state_sp.compute(state, learn=learn)
        if self.state_clusters is not None:
            nom = self.state_clusters.number_of_clusters
            # if nom > 45:
            #     self.state_clusters = Memory(
            #         self.state_clusters.size, self.state_clusters.threshold
            #     )

            self.state_clusters.add(s)
            s, _ = self._match_nearest_cluster(
                s, self.state_clusters, self.state_sp.n_active_bits
            )
            self._increase_threshold(
                self.state_clusters, self.state_clusters_similarity_threshold_max
            )
            nom_ = self.state_clusters.number_of_clusters
            # if nom_ > nom and nom_ % 10 == 0:
            #     print(nom_)
        return s

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
        sa = self.sa_sp.compute(s_a, learn=learn)
        return sa

    def concat_s_action(self, s: SparseSdr, action: int, learn: bool) -> SparseSdr:
        a = self.encode_action(action, learn=learn)
        s_a = self.concat_s_a(s, a, learn=learn)
        return s_a

    def encode_s_action(self, s: SparseSdr, action: int, learn: bool) -> SparseSdr:
        s_a = self.concat_s_action(s, action, learn=learn)
        sa = self.encode_s_a(s_a, learn=learn)
        return sa

    @property
    def output_sdr_size(self):
        return self.sa_sp.output_sdr_size

    @property
    def s_output_sdr_size(self):
        return self.state_sp.output_sdr_size

    def restore_s(self, s: SparseSdr, threshold: float):
        if len(s) / self.state_sp.n_active_bits < threshold:
            return s
        if self.state_clusters is None:
            return s

        restored_s, similarity = self._match_nearest_cluster(
            s, self.state_clusters, self.state_sp.n_active_bits
        )
        if similarity < threshold:
            return s
        return restored_s

    def match_nearest_cluster(self, sdr: SparseSdr):
        clusters = self.state_clusters
        n_active_bits = self.state_sp.n_active_bits

        dense_sdr = sparse_to_dense(sdr, clusters.size)
        similarities = clusters.similarity(dense_sdr)
        best_match_cluster_ind = np.argmax(similarities)

        sparsity = n_active_bits / clusters.size
        cluster_norms = clusters.adopted_kernels(sparsity)
        best_cluster_norm = cluster_norms[best_match_cluster_ind]
        best_cluster = np.flatnonzero(best_cluster_norm)
        return best_match_cluster_ind, similarities, best_cluster

    @staticmethod
    def _match_nearest_cluster(
            sdr: SparseSdr, clusters: Memory, n_active_bits: int
    ) -> SparseSdr:
        dense_sdr = sparse_to_dense(sdr, clusters.size)
        similarities = clusters.similarity(dense_sdr)
        best_match_cluster_ind = np.argmax(similarities)

        sparsity = n_active_bits / clusters.size
        cluster_norms = clusters.adopted_kernels(sparsity)
        best_cluster_norm = cluster_norms[best_match_cluster_ind]
        best_cluster = np.flatnonzero(best_cluster_norm)
        return best_cluster, similarities[best_match_cluster_ind]

    @staticmethod
    def _increase_threshold(clusters: Memory, max_threshold: float):
        if clusters.threshold < max_threshold:
            clusters.threshold += .0004


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
        # noinspection PyUnusedLocal
        sa = s_a = self.sa_encoder.encode(s, action)
        return sa

    @property
    def s_output_sdr_size(self):
        return self.sa_encoder.input_sdr_size

    @property
    def output_sdr_size(self):
        return self.sa_encoder.output_sdr_size


def make_sa_encoder(
        env: Env, seed: int, sa_encoder_config: dict
):
    if sa_encoder_config:
        return SpSaEncoder(env, seed, **sa_encoder_config)
    else:
        return CrossSaEncoder(env)
