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
