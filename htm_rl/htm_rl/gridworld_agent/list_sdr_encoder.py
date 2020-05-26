from itertools import chain
from typing import List, NamedTuple, Iterable

from htm_rl.gridworld_agent.sar import SuperpositionList
from htm_rl.representations.int_sdr_encoder import IntSdrEncoder, BitRange
from htm_rl.representations.sdr import SparseSdr
from htm_rl.utils import isnone

Dim2d = NamedTuple('Dim2d', (('rows', int), ('cols', int)))


class ListSdrEncoder:
    value_bits: int
    total_bits: int
    activation_threshold: int

    _encoder: IntSdrEncoder
    _n_elems: int
    _n_dim: Dim2d
    _shifts: List[int]

    def __init__(self, encoder: IntSdrEncoder, n_dim: Dim2d):
        n_rows, n_cols = n_dim
        n_elems = n_rows * n_cols

        self.value_bits = n_elems * encoder.value_bits
        self.total_bits = n_elems * encoder.total_bits
        self.activation_threshold = n_elems * encoder.activation_threshold

        self._encoder = encoder
        self._n_elems = n_elems
        self._n_dim = n_dim
        self._shifts = self._get_shifts(encoder, n_elems)

    def encode(self, values: List[int], base_shift: int) -> Iterable[BitRange]:
        values = isnone(values, [])
        return (
            bit_range
            for x, shift in zip(values, self._shifts)
            for bit_range in self._encoder.encode(x, base_shift + shift)
        )

    def decode(self, indices: SparseSdr) -> SuperpositionList:
        elem_sdrs = self._split_by_list_elems(indices)
        return list(map(self._encoder.decode, elem_sdrs))

    def format(self, indices: SparseSdr) -> str:
        elem_sdrs = self._split_by_list_elems(indices)
        rows, cols = self._n_dim

        return '\n'.join(
            ' | '.join(
                self._encoder.format(elem_sdrs[row * cols + col])
                for col in range(cols)
            )
            for row in range(rows)
        )

    def _split_by_list_elems(self, indices: SparseSdr) -> List[SparseSdr]:
        buckets: List[SparseSdr] = [[] for _ in range(self._n_elems)]
        for ind in indices:
            elem, intra_elem_ind = divmod(ind, self._encoder.total_bits)
            buckets[elem].append(intra_elem_ind)
        return buckets

    @staticmethod
    def _get_shifts(encoder: IntSdrEncoder, n_elems: int) -> List[int]:
        step = encoder.total_bits
        return list(range(0, step * n_elems, step))
