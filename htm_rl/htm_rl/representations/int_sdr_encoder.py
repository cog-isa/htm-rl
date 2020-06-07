from dataclasses import dataclass, astuple
from typing import Iterable

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
    n_values: int
    value_bits: int
    total_bits: int
    activation_threshold: int

    def __init__(self, name: str, n_values: int, value_bits: int, activation_threshold: int = None):
        self.name = name
        self.n_values = n_values
        self.value_bits = value_bits
        self.total_bits = n_values * value_bits
        self.activation_threshold = isnone(activation_threshold, value_bits)

    def encode(self, x: int, shift: int) -> Iterable[BitRange]:
        l, r = self._bit_bucket_range(x)
        yield BitRange(l + shift, r + shift)

    def decode(self, indices: Iterable[int]) -> Superposition:
        n_activations = [0] * self.n_values
        for x in indices:
            value = x // self.value_bits
            n_activations[value] += 1

        decoded_values = [
            value
            for value, n_act in enumerate(n_activations)
            if n_act >= self.activation_threshold
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

        l = x * self.value_bits
        r = l + self.value_bits
        return BitRange(l, r)

    def _str_from_dense(self, arr: DenseSdr) -> str:
        bit_buckets = (
            arr[i: i + self.value_bits]
            for i in range(0, self.total_bits, self.value_bits)
        )
        return ' '.join(
            ''.join(map(str, bucket))
            for bucket in bit_buckets
        )

    def _assert_acceptable_values(self, x: int):
        assert x is None or x == self.ALL or 0 <= x < self.n_values, \
            f'Value must be in [0, {self.n_values}] or {self.ALL} or None; got {x}'

    def __str__(self):
        name, n_values, value_bits = self.name, self.n_values, self.value_bits
        return f'({name}: v{n_values} x b{value_bits})'


class IntSdrEncoderShortFormat(IntSdrEncoder):
    def __init__(self, name: str, n_values: int, value_bits: int, activation_threshold: int = None):
        super().__init__(name, n_values, value_bits, activation_threshold)

    def format(self, indices: SparseSdr) -> str:
        n_activations = [0] * self.n_values
        for x in indices:
            value = x // self.value_bits
            n_activations[value] += 1
        
        return ' '.join(
            '+' if n_active_bits == self.value_bits else '.' if n_active_bits > 0 else '-'
            for n_active_bits in value_active_bits
        )
