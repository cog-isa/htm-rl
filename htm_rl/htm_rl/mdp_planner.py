from typing import List, Tuple

import numpy as np
from htm.bindings.sdr import SDR
from htm.algorithms import TemporalMemory as TM, Connections


class DataEncoder:
    activation_threshold: int
    input_size: int
    value_bits: int
    n_vals: int

    def __init__(self, value_bits: int, n_vals: int, activation_threshold: int = None):
        self.value_bits = value_bits
        self.n_vals = n_vals
        self.input_size = n_vals * value_bits
        self.activation_threshold = isnone(activation_threshold, self.value_bits)

    def encode_sparse(self, x: int) -> List[int]:
        ind_from, ind_to = self._get_encode_range(x)
        return list(range(ind_from, ind_to))

    def encode_dense(self, x: int, arr: np.array = None) -> np.array:
        ind_from, ind_to = self._get_encode_range(x)
        if arr is None:
            arr = np.zeros(self.input_size, dtype=np.int8)
        arr[ind_from:ind_to] = 1
        return arr

    def _get_encode_range(self, x: int) -> Tuple[int, int]:
        assert x < self.n_vals, f'Value must be in [0, {self.n_vals}]; got {x}'
        ind_from = x * self.value_bits
        ind_to = ind_from + self.value_bits
        return ind_from, ind_to

    def str_from_dense(self, arr: np.array):
        assert arr.ndim == 1, f'Expected 1-dim array; got {arr.ndim}'
        bit_packs_by_value = [
            ''.join(map(str, arr[i: i + self.value_bits]))
            for i in range(0, self.input_size, self.value_bits)
        ]
        return ' '.join(bit_packs_by_value)

    def decode_dense(self, arr: np.array) -> List[int]:
        res = []
        for i in range(0, self.input_size, self.value_bits):
            s = np.sum(arr[i: i + self.value_bits])
            if s >= self.activation_threshold:
                res.append(i // self.value_bits)
        return res

    def decode_sparse(self, arr: List[int]) -> List[int]:
        activations = [0]*self.n_vals
        for x in arr:
            activations[x // self.value_bits] += 1
        return [i for i, act in enumerate(activations) if act >= self.activation_threshold]


class TestDataEncoder:
    def __init__(self):
        self.encoder = DataEncoder(value_bits=3, n_vals=2, activation_threshold=2)

    def test_encode_sparse(self):
        arr_sparse = self.encoder.encode_sparse(1)
        assert arr_sparse == [3, 4, 5]

    def test_str_from_dense(self):
        arr_dense = np.array([0, 0, 0, 1, 1, 1], dtype=np.int8)
        res = self.encoder.str_from_dense(arr_dense)
        assert res == '000 111'

    def test_decode_dense(self):
        decoded = self.encoder.decode_dense(np.array([0, 1, 1, 0, 1, 0]))
        assert decoded == [0]

        decoded = self.encoder.decode_dense(np.array([0, 1, 1, 1, 1, 1]))
        assert decoded == [0, 1]

    def test_decode_sparse(self):
        decoded = self.encoder.decode_sparse([1, 2, 4])
        assert decoded == [0]

        decoded = self.encoder.decode_sparse([1, 2, 3, 4, 5])
        assert decoded == [0, 1]





def main():
    encoder_tester = TestDataEncoder()
    encoder_tester.test_encode_sparse()
    encoder_tester.test_str_from_dense()


def isnone(x, default):
    return x if x is not None else default


main()

