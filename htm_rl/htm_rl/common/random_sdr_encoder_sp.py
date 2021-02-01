from typing import List, Iterable

import numpy as np

from htm_rl.common.base_sa import Superposition
from htm_rl.common.random_sdr_encoder import RandomSdrEncoder, BitArr
from htm_rl.common.sdr import SparseSdr
from htm_rl.htm_plugins.spatial_pooler import SpatialPooler


class BitSparseArr:
    """
    Opposite to bit range
    """
    sparse_sdr: SparseSdr
    _shift: int

    def __init__(self, sparse_sdr, shift=0):
        self.sparse_sdr = sparse_sdr
        self._shift = shift

    # is needed for tuple unpacking
    def __iter__(self):
        yield from self.unfold()

    def shift(self, shift: int) -> 'BitSparseArr':
        return BitSparseArr(self.sparse_sdr, self._shift + shift)

    def unfold(self) -> Iterable[int]:
        """Materialize range into actual indices."""
        for x in self.sparse_sdr:
            yield x + self._shift


class RandomSdrEncoderSp:
    random_encoder: RandomSdrEncoder
    spatial_pooler: SpatialPooler
    name: str
    value_bits: int
    total_bits: int
    activation_threshold: int

    def __init__(
            self, name: str, random_encoder: RandomSdrEncoder, spatial_pooler: SpatialPooler
    ):
        self.name = name
        self.random_encoder = random_encoder
        self.spatial_pooler = spatial_pooler

        sparsity = self.spatial_pooler.spatial_pooler.getLocalAreaDensity()
        total_bits = self.spatial_pooler.spatial_pooler.getColumnDimensions()[0]
        self.total_bits = total_bits
        self.value_bits = int(sparsity * total_bits)
        self.activation_threshold = int(.8 * self.value_bits)

    def encode(self, x) -> BitSparseArr:
        if self.random_encoder is not None:
            sparse_sdr = list(self.random_encoder.encode(x).unfold())
        else:
            sparse_sdr = list(np.nonzero(x)[0])
        sparse_sdr = self.spatial_pooler.encode(sparse_sdr)
        return BitSparseArr(sparse_sdr)

    def decode(self, sparse_sdr: SparseSdr) -> Superposition:
        return []

    def has_value(self, sparse_sdr: SparseSdr, value: int) -> bool:
        assert False, 'Not implemented'

    def value_activations(self, sparse_sdr: SparseSdr) -> List[int]:
        assert False, 'Not implemented'
        return []

    def format(self, sparse_sdr: SparseSdr, format_: str = None) -> str:
        return self.random_encoder.format(sparse_sdr, format_)

    def __str__(self):
        assert False, 'Not implemented'
