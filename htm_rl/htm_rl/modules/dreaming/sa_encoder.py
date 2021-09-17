import numpy as np

from htm_rl.agents.q.sa_encoders import SpSaEncoder
from htm_rl.common.sdr import SparseSdr
from htm_rl.common.sdr_encoders import IntBucketEncoder, SdrConcatenator
from htm_rl.htm_plugins.spatial_pooler import SpatialPoolerWrapper
from htm_rl.modules.empowerment import Memory


class DreamerSaEncoder(SpSaEncoder):
    """
    Auxiliary encoder for the HIMA dreamer implementation.
    Besides forward encoding, it also supports s --> state decoding. It also
    doesn't implement s_a --> sa encoding as it's needed for a Dreamer.
    """
    sa_sp: None

    state_clusters: Memory
    state_decoder: list[SparseSdr]

    # noinspection PyMissingConstructor
    def __init__(
            self, state_encoder, n_actions: int,
            clusters_similarity_threshold: float
    ):
        self.state_sp = SpatialPoolerWrapper(state_encoder)
        self.state_clusters = Memory(
            size=self.state_sp.output_sdr_size,
            threshold=clusters_similarity_threshold
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
        s = super(DreamerSaEncoder, self).encode_state(state, learn=learn)

        self._add_to_decoder(state, s)
        return s

    def decode_s_to_state(self, s: SparseSdr) -> SparseSdr:
        similarity_with_clusters = self.state_clusters.similarity(s)
        i_state_cluster = np.argmax(similarity_with_clusters)
        return self.state_decoder[i_state_cluster]

    @property
    def output_sdr_size(self):
        return self.s_a_concatenator.output_sdr_size

    def _add_to_decoder(self, state: SparseSdr, s: SparseSdr):
        similarity_with_clusters = self.state_clusters.similarity(s)
        if similarity_with_clusters.size > 0:
            i_state_cluster = np.argmax(similarity_with_clusters)
            if i_state_cluster < len(self.state_decoder):
                assert np.all(state == self.state_decoder[i_state_cluster])
            else:
                self.state_decoder.append(state.copy())
