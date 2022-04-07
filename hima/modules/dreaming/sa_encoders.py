from typing import Optional

import numpy as np

from hima.modules.dreaming.cluster_memory import ClusterMemory
from hima.modules.dreaming.sa_encoder import SaEncoder
from hima.common.sdr import SparseSdr
from hima.common.sdr_encoders import IntBucketEncoder, SdrConcatenator
from hima.envs.env import Env
from hima.modules.htm.spatial_pooler import SpatialPoolerWrapper, SpatialPooler


class SpSaEncoder(SaEncoder):
    state_sp: SpatialPooler
    action_encoder: IntBucketEncoder
    s_a_concatenator: SdrConcatenator
    sa_sp: SpatialPooler

    state_clusters: Optional[ClusterMemory]
    state_clusters_similarity_threshold_max: Optional[float]

    def __init__(
            self,
            env: Env,
            seed: int,
            state_sp: dict,
            action_encoder: dict,
            sa_sp: dict,
            state_clusters: dict = None
    ):
        self.state_sp = SpatialPooler(input_source=env, seed=seed, **state_sp)
        self.state_clusters = None
        if state_clusters is not None:
            self.state_clusters = ClusterMemory(
                input_sdr_size=self.s_output_sdr_size,
                n_active_bits=self.state_sp.n_active_bits,
                **state_clusters
            )

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
            s = self._cluster_s(s, learn)
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

    def restore_s(self, s: SparseSdr):
        restoration_threshold = self.state_clusters.similarity_threshold.min_value

        # if s damaged too much, just don't try restoring
        if len(s) / self.state_sp.n_active_bits < restoration_threshold:
            return s
        if self.state_clusters is None:
            return s

        cluster, i_cluster, similarity = self.state_clusters.match(
            s, similarity_threshold=restoration_threshold
        )
        if cluster is not None:
            return np.sort(cluster)
        return s

    def _cluster_s(self, s: SparseSdr, learn: bool) -> SparseSdr:
        cluster, i_cluster, similarity = self.state_clusters.match(s)
        self.state_clusters.similarity_threshold.balance(increase=cluster is not None)

        if learn:
            i_cluster = self.state_clusters.activate(
                s, similarity=similarity, matched_cluster=i_cluster
            )
            cluster = self.state_clusters.representatives[i_cluster]

        if cluster is not None:
            s = np.sort(cluster)

        return s


class DreamerSaEncoder(SpSaEncoder):
    """
    Auxiliary encoder for the HIMA dreamer implementation.
    Besides forward encoding, it also supports s --> state decoding. It also
    doesn't implement s_a --> sa encoding as it's not needed for a Dreamer.
    """
    sa_sp: None

    state_sp: SpatialPoolerWrapper
    state_clusters: ClusterMemory
    state_decoder: list[SparseSdr]

    # noinspection PyMissingConstructor
    def __init__(
            self, state_encoder, n_actions: int,
            state_clusters: dict
    ):
        self.state_sp = SpatialPoolerWrapper(state_encoder)
        self.state_clusters = ClusterMemory(
            input_sdr_size=self.state_sp.output_sdr_size,
            n_active_bits=self.state_sp.n_active_bits,
            **state_clusters
        )
        self.state_decoder = []

        self.action_encoder = IntBucketEncoder(
            n_actions, self.state_sp.n_active_bits
        )
        self.s_a_concatenator = SdrConcatenator(input_sources=[
            self.state_sp,
            self.action_encoder
        ])
        self.sa_sp = None  # it isn't needed for a Dreamer

    def encode_state(self, state: SparseSdr, learn: bool) -> SparseSdr:
        if not isinstance(state, np.ndarray):
            state = np.array(list(state))

        # state encoder learns in HIMA, not here
        s = self.state_sp.compute(state, learn=False)

        s, i_cluster = self._cluster_s(s, learn)
        self._add_to_decoder(state, i_cluster)
        return s

    def concat_s_action(self, s: SparseSdr, action: int, learn: bool) -> SparseSdr:
        # action encoder learns in HIMA, not here
        a = self.encode_action(action, learn=False)
        s_a = self.concat_s_a(s, a, learn=learn)
        return s_a

    def decode_s_to_state(self, s: SparseSdr) -> SparseSdr:
        cluster, i_cluster, similarity = self.state_clusters.match(
            s,
            similarity_threshold=self.state_clusters.similarity_threshold.min_value
        )
        if cluster is None:
            return np.empty(0)
        return self.state_decoder[i_cluster]

    @property
    def output_sdr_size(self):
        return self.s_a_concatenator.output_sdr_size

    def _cluster_s(self, s: SparseSdr, learn: bool) -> SparseSdr:
        cluster, i_cluster, similarity = self.state_clusters.match(s)
        self.state_clusters.similarity_threshold.balance(increase=cluster is not None)

        if learn:
            if cluster is None and self.state_clusters.full:
                # have to free up space manually to track clusters order change
                i_removed = self.state_clusters.remove_least_used_cluster()

                self.state_decoder[i_removed] = self.state_decoder[-1]
                self.state_decoder.pop()

            i_cluster = self.state_clusters.activate(
                s, similarity=similarity, matched_cluster=i_cluster
            )
            cluster = self.state_clusters.representatives[i_cluster]

        if cluster is not None:
            s = np.sort(cluster)

        return s, i_cluster

    def _add_to_decoder(self, state: SparseSdr, cluster: int):
        if cluster is None:
            return

        if cluster == len(self.state_decoder):
            self.state_decoder.append([])
        self.state_decoder[cluster] = state.copy()
