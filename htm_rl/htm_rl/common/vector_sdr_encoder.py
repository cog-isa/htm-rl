from typing import Tuple, Iterable, List

import numpy as np

from htm_rl.common.base_sa import Superposition
from htm_rl.common.random_sdr_encoder import BitArr
from htm_rl.common.sdr import SparseSdr, dense_to_sparse, sparse_to_dense
from htm_rl.common.utils import isnone, trace


class VectorSdrEncoder:
    name: str
    value_bits: int
    total_bits: int
    activation_threshold: int
    encoding_map: np.array
    fixed_base: Tuple[int, int]

    def __init__(
            self, name: str, n_elements: int, n_categories: int,
            default_format: str = 'full'
    ):
        self.name = name
        self.n_elements = n_elements
        self.n_categories = n_categories
        self._cache_vector = np.zeros((n_elements, n_categories), dtype=np.int8)

        self.total_bits = n_elements * n_categories
        # expected number of value bits
        self.value_bits = n_elements
        self.activation_threshold = int(.9 * self.value_bits)
        self.default_format = default_format

    def encode(self, x: np.array) -> BitArr:
        self._cache_vector.fill(0)
        self._cache_vector[np.arange(self.n_elements), x] = 1
        dense_sdr = self._cache_vector.ravel()
        return BitArr(dense_sdr)

    def decode(self, sparse_sdr: SparseSdr) -> Superposition:
        dense_sdr = sparse_to_dense(sparse_sdr, self.total_bits)
        dense_sdr = dense_sdr.reshape((self.n_elements, self.n_categories))
        idx, cat = np.nonzero(dense_sdr)
        obs = np.zeros(self.n_elements, dtype=np.int8)
        obs[idx] = cat
        return [obs]

    def has_value(self, sparse_sdr: SparseSdr, value: int) -> bool:
        assert False, 'Not implemented'
        return False

    def value_activations(self, sparse_sdr: SparseSdr) -> List[int]:
        """
        Counts number of activations in each value's bucket of bits in a given sparse SDR.
        :param sparse_sdr: sparse SDR
        :return: list of activations - i-th elem is an activation of value i.
        """
        assert False, 'Not implemented'
        return []

    def format(self, sparse_sdr: SparseSdr, format_: str = None) -> str:
        """
        Formats sparse SDR to string with one of the supported formats.

        Supported formats are: 'full' and 'short'. If None then encoder's default is used.
        """
        format_ = isnone(format_, self.default_format)
        supported_formats = {
            'full': VectorSdrFormatter
        }
        return supported_formats[format_].format(sparse_sdr, self)

    def __str__(self):
        name, value_bits = self.name, self.value_bits
        return f'({name}: b{value_bits})'


class VectorSdrFormatter:
    """
    Formats SDR as a dense array:
        0000 1111 0000
    """

    @staticmethod
    def format(sparse_sdr: SparseSdr, encoder: VectorSdrEncoder) -> str:
        """
        :param sparse_sdr: SDR to print into string
        :param encoder: encoder for this particular kind of SDRs
        """
        dense_sdr = sparse_to_dense(sparse_sdr, encoder.total_bits)
        return ''.join(map(str, dense_sdr))
