from typing import List, Tuple, Iterable

import numpy as np

from representations.sdr import SparseSdr, DenseSdr
from utils import isnone, sparse_to_dense


class IntSdrEncoder:
    ALL = -1

    name: str
    n_values: int
    value_bits: int
    total_bits: int
    activation_threshold: int

    def __init__(self, name: str, n_vals: int, value_bits: int, activation_threshold: int = None):
        self.name = name
        self.value_bits = value_bits
        self.n_values = n_vals
        self.total_bits = n_vals * value_bits
        self.activation_threshold = isnone(activation_threshold, self.value_bits)

    def encode_sparse_with_shift(self, x: int, shift: int) -> Iterable[int]:
        ind_from, ind_to = self._encoding_bits_range(x)
        return range(ind_from + shift, ind_to + shift)

    def encode_sparse(self, x: int) -> SparseSdr:
        ind_from, ind_to = self._encoding_bits_range(x)
        return list(range(ind_from, ind_to))

    def _encoding_bits_range(self, x: int) -> Tuple[int, int]:
        assert x is None or x == self.ALL or 0 <= x < self.n_values, \
            f'Value must be in [0, {self.n_values}] or {self.ALL} or None; got {x}'

        if x is None:
            return 0, 0
        if x == self.ALL:
            return 0, self.total_bits

        ind_from = x * self.value_bits
        ind_to = ind_from + self.value_bits
        return ind_from, ind_to

    def decode_sparse(self, indices: SparseSdr) -> List[int]:
        n_activations = [0]*self.n_values
        for x in indices:
            value = x // self.value_bits
            n_activations[value] += 1

        decoded_values = [
            value
            for value, n_act in enumerate(n_activations)
            if n_act >= self.activation_threshold
        ]
        return decoded_values

    def str_from_sparse(self, indices: SparseSdr):
        return self._str_from_dense(sparse_to_dense(indices, self.total_bits))

    def _encode_dense(self, x: int, arr: DenseSdr = None) -> DenseSdr:
        ind_from, ind_to = self._encoding_bits_range(x)
        if arr is None:
            arr = np.zeros(self.total_bits, dtype=np.int8)
        arr[ind_from:ind_to] = 1
        return arr

    def _decode_dense(self, arr: DenseSdr) -> List[int]:
        decoded_values = []
        for i in range(0, self.total_bits, self.value_bits):
            n_activations = np.count_nonzero(arr[i: i + self.value_bits])
            if n_activations >= self.activation_threshold:
                decoded_values.append(i // self.value_bits)

        return decoded_values

    def _str_from_dense(self, arr: DenseSdr):
        assert arr.ndim == 1, f'Expected 1-dim array; got {arr.ndim}'
        bit_packs_by_value = [
            ''.join(map(str, arr[i: i + self.value_bits]))
            for i in range(0, self.total_bits, self.value_bits)
        ]
        return ' '.join(bit_packs_by_value)

    def __str__(self):
        name, n_values, value_bits = self.name, self.n_values, self.value_bits
        return f'({name}: v{n_values} x b{value_bits})'
