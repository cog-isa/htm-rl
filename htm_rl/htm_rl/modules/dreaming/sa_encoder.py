import numpy as np

from htm_rl.agents.q.cluster_memory import ClusterMemory
from htm_rl.agents.q.sa_encoders import SpSaEncoder
from htm_rl.common.sdr import SparseSdr
from htm_rl.common.sdr_encoders import IntBucketEncoder, SdrConcatenator
from htm_rl.htm_plugins.spatial_pooler import SpatialPoolerWrapper


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
        self.state_clusters.adapt_similarity_threshold(is_hit=cluster is not None)

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
