from dataclasses import dataclass, astuple
from typing import Tuple, Iterable

from htm_rl.mdp_agent.sar import Superposition
from htm_rl.representations.sdr import SparseSdr, DenseSdr
from htm_rl.utils import isnone, sparse_to_dense


@dataclass(frozen=True)
class BitRange:
    __slots__ = ['l', 'r']
    l: int
    r: int

    # is needed for tuple unpacking
    def __iter__(self):
        yield from astuple(self)

    @staticmethod
    def unfold(buckets: Iterable['BitRange']) -> SparseSdr:
        return [
            i
            for bucket in buckets
            for i in range(bucket.l, bucket.r)
        ]


class IntSdrEncoder:
    ALL = -1

    name: str
    _n_values: int
    _value_bits: int
    total_bits: int
    _activation_threshold: int

    def __init__(self, name: str, n_values: int, value_bits: int, activation_threshold: int = None):
        self._value_bits = value_bits
        self._n_values = n_values
        self._activation_threshold = isnone(activation_threshold, self._value_bits)

        self.name = name
        self.total_bits = n_values * value_bits

    def encode(self, x: int, shift: int) -> Iterable[BitRange]:
        l, r = self._bit_bucket_range(x)
        yield BitRange(l + shift, r + shift)

    def decode(self, indices: Iterable[int]) -> Superposition:
        n_activations = [0] * self._n_values
        for x in indices:
            value = x // self._value_bits
            n_activations[value] += 1

        decoded_values = [
            value
            for value, n_act in enumerate(n_activations)
            if n_act >= self._activation_threshold
        ]
        return decoded_values

    def format(self, indices: SparseSdr) -> str:
        dense_sdr = sparse_to_dense(indices, self.total_bits)
        return self._str_from_dense(dense_sdr)

    def _bit_bucket_range(self, x: int) -> BitRange:
        self._assert_acceptable_values(x)

        if x is None:
            return BitRange(0, 0)
        if x == self.ALL:
            return BitRange(0, self.total_bits)

        l = x * self._value_bits
        r = l + self._value_bits
        return BitRange(l, r)

    def _str_from_dense(self, arr: DenseSdr) -> str:
        bit_buckets = (
            arr[i: i + self._value_bits]
            for i in range(0, self.total_bits, self._value_bits)
        )
        return ' '.join(
            ''.join(map(str, bucket))
            for bucket in bit_buckets
        )

    def _assert_acceptable_values(self, x: int):
        assert x is None or x == self.ALL or 0 <= x < self._n_values, \
            f'Value must be in [0, {self._n_values}] or {self.ALL} or None; got {x}'

    def __str__(self):
        name, n_values, value_bits = self.name, self._n_values, self._value_bits
        return f'({name}: v{n_values} x b{value_bits})'


# class IntSdrEncoderBackup:
#     ALL = -1
#
#     name: str
#     n_values: int
#     value_bits: int
#     total_bits: int
#     activation_threshold: int
#
#     def __init__(self, name: str, n_vals: int, value_bits: int, activation_threshold: int = None):
#         self.name = name
#         self.value_bits = value_bits
#         self.n_values = n_vals
#         self.total_bits = n_vals * value_bits
#         self.activation_threshold = isnone(activation_threshold, self.value_bits)
#
#     def encode_chained(self, x: int, shift: int) -> Iterable[int]:
#         l, r = self._bit_bucket_range(x)
#         return range(l + shift, r + shift)
#
#     def decode_chained(self, indices: Iterable[int], shift: int) -> Superposition[int]:
#         n_activations = [0] * self.n_values
#         for x in indices:
#             value = (x - shift) // self.value_bits
#             n_activations[value] += 1
#
#         decoded_values = [
#             value
#             for value, n_act in enumerate(n_activations)
#             if n_act >= self.activation_threshold
#         ]
#         return decoded_values
#
#     def encode(self, x: int) -> SparseSdr:
#         return list(self.encode_chained(x, 0))
#
#     def decode(self, indices: SparseSdr) -> Superposition[int]:
#         return self.decode_chained(indices, 0)
#
#     def to_str(self, indices: SparseSdr):
#         dense_sdr = sparse_to_dense(indices, self.total_bits)
#         return self._str_from_dense(dense_sdr)
#
#     def __str__(self):
#         name, n_values, value_bits = self.name, self.n_values, self.value_bits
#         return f'({name}: v{n_values} x b{value_bits})'
#
#     def _bit_bucket_range(self, x: int) -> Tuple[int, int]:
#         self._assert_acceptable_values(x)
#
#         if x is None:
#             return 0, 0
#         if x == self.ALL:
#             return 0, self.total_bits
#
#         ind_from = x * self.value_bits
#         ind_to = ind_from + self.value_bits
#         return ind_from, ind_to
#
#     def _str_from_dense(self, arr: DenseSdr):
#         assert arr.ndim == 1, f'Expected 1-dim array; got {arr.ndim}'
#         bit_buckets = (
#             arr[i: i + self.value_bits]
#             for i in range(0, self.total_bits, self.value_bits)
#         )
#         return ' '.join(
#             ''.join(map(str, bucket))
#             for bucket in bit_buckets
#         )
#
#     def _assert_acceptable_values(self, x: int):
#         assert x is None or x == self.ALL or 0 <= x < self.n_values, \
#             f'Value must be in [0, {self.n_values}] or {self.ALL} or None; got {x}'
#
