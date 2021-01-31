from typing import Tuple, Iterable, List

import numpy as np

from htm_rl.common.base_sa import Superposition
from htm_rl.common.sdr import SparseSdr, dense_to_sparse, sparse_to_dense
from htm_rl.common.utils import isnone, trace


class BitArr:
    """
    Opposite to bit range
    """
    arr: np.array

    def __init__(self, arr):
        self.arr = arr

    # is needed for tuple unpacking
    def __iter__(self):
        yield from dense_to_sparse(self.arr)

    def shift(self, shift: int) -> 'BitArr':
        return BitArr(self.arr + shift)

    def unfold(self) -> Iterable[int]:
        """Materialize range into actual indices."""
        yield from dense_to_sparse(self.arr)


class RandomSdrEncoder:
    """
    Encodes integer values from range [0, `n_values`) as SDR w/ `total_bits` bits.
    Every single value x from the range is encoded as `value_bits` contiguous 1 bits called
        a bucket. All bits outside corresponding bucket are set to zero.

    Example:
        An encoder(n_values = 3, value_bits = 4) encodes value 1 as:
            0000 1111 0000
        Buckets are separated by empty space for clarity. Note, that there're 3 buckets of size 4.
    """
    name: str
    value_bits: int
    total_bits: int
    activation_threshold: int
    encoding_map: np.array
    fixed_base: Tuple[int, int]

    def __init__(
            self, name: str, n_states: int, fixed_base: Tuple[int, int], total_bits: int,
            sparsity: float, seed: int, default_format: str = 'full'
    ):
        """
        Initializes encoder.

        :param name: a name for values encoded
        :param n_values: defines a range [0, n_values) of values that can be encoded
        :param value_bits: a number of bits in a bucket used to encode single value
        :param activation_threshold: a number of 1 bits in a bucket, which are enough
            to activate corresponding value, i.e. this value will be decoded.
        :param default_format: printing format
        """
        self.name = name
        self.encoding_map = np.zeros(shape=(n_states, total_bits), dtype=np.int8)
        self.fixed_base = fixed_base

        rng_generator = np.random.default_rng(seed=seed)
        self._init_encoding_map(self.encoding_map, fixed_base, sparsity, rng_generator)
        self.total_bits = total_bits
        # expected number of value bits
        self.value_bits = fixed_base[1] + int((total_bits - fixed_base[0]) * sparsity)
        self.activation_threshold = int(.85 * self.value_bits)
        self.default_format = default_format

    def _init_encoding_map(self, encoding_map, fixed_base, sparsity, rng_generator):
        # init fixed base
        fixed_base_total, fixed_base_ones = fixed_base
        encoding_map[:, 0:fixed_base_ones] = 1

        n_states, total_bits = encoding_map.shape
        choose_k_bits = int((total_bits - fixed_base_total) * sparsity)
        for state in range(n_states):
            bits = rng_generator.choice(total_bits-fixed_base_total, choose_k_bits, replace=False)
            encoding_map[state, bits] = 1

        # DEBUG part: calculate intersection stats
        shared_cells = np.array([
            np.count_nonzero(encoding_map[s1] * encoding_map[s2])
            for s1 in range(n_states)
            for s2 in range(n_states)
        ]).reshape((n_states, n_states))
        shared_cells = shared_cells - np.eye(n_states) * choose_k_bits

        print((shared_cells.mean(axis=1)).astype(np.int))
        print((shared_cells.max(axis=1)).astype(np.int))
        # DEBUG part

    def encode(self, x: int) -> BitArr:
        dense_sdr = self.encoding_map[x]
        return BitArr(dense_sdr)

    def decode(self, sparse_sdr: SparseSdr) -> Superposition:
        """
        Decodes sparse SDR into superposition of single values.
        Note that SDR may contain multiple single values, that's why the result is not a single value,
            but a superposition of them (represented as list of single values).
        A single value is "decoded" if its corresponding bucket of bits has
            enough, i.e. >= activation_threshold, 1 bits.
        :param sparse_sdr: sparse SDR to decode
        :return: list of decoded values.
        """
        matches = np.count_nonzero(self.encoding_map[:, sparse_sdr], axis=1)
        decoded = matches >= self.activation_threshold
        states = np.nonzero(decoded)[0]
        return states

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
            'full': RandomSdrFormatter
        }
        return supported_formats[format_].format(sparse_sdr, self)

    def __str__(self):
        name, value_bits = self.name, self.value_bits
        return f'({name}: b{value_bits})'


class RandomSdrFormatter:
    """
    Formats SDR as a dense array:
        0000 1111 0000
    """

    @staticmethod
    def format(sparse_sdr: SparseSdr, encoder: RandomSdrEncoder) -> str:
        """
        :param sparse_sdr: SDR to print into string
        :param encoder: encoder for this particular kind of SDRs
        """
        dense_sdr = sparse_to_dense(sparse_sdr, encoder.total_bits)
        value_bits = dense_sdr[encoder.fixed_base[0]:]
        return ''.join(map(str, value_bits))
