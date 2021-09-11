from typing import List

from htm_rl.common.sdr import SparseSdr


class SaEncoder:
    def encode_state(self, state: SparseSdr, learn: bool) -> SparseSdr:
        raise NotImplementedError()

    def encode_actions(self, s: SparseSdr, learn: bool) -> List[SparseSdr]:
        raise NotImplementedError()

    def encode_sa(self, s: SparseSdr, action: int, learn: bool) -> SparseSdr:
        raise NotImplementedError()

    def decode_state(self, sdr: SparseSdr) -> SparseSdr:
        raise NotImplementedError()

    @property
    def output_sdr_size(self):
        raise NotImplementedError()