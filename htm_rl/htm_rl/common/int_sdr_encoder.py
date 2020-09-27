from dataclasses import dataclass, astuple
from typing import Iterable, List

from htm_rl.common.base_sar import Superposition
from htm_rl.common.sdr import SparseSdr, sparse_to_dense
from htm_rl.common.utils import isnone
from typing import TypeVar, Generic
from itertools import product, count

import numpy as np

ValueEncoder = TypeVar('ValueEncoder')
SequenceEncoder = TypeVar('SequenceEncoder')


@dataclass(frozen=True)
class BitRange:
    """
    Shorthand for representing contiguous index range [l, r) of 1 bits.
    Used to represent an active bucket during encoding. Use `unfold` method to get actual set of indices.
    """
    __slots__ = ['l', 'r']
    l: int
    r: int

    # is needed for tuple unpacking
    def __iter__(self):
        yield from astuple(self)

    def shift(self, shift: int) -> 'BitRange':
        return BitRange(self.l + shift, self.r + shift)

    def unfold(self) -> Iterable[int]:
        """Materialize range into actual indices."""
        return range(self.l, self.r)


class BitRangePack:
    """
        Shorthand for representing contiguous index ranges [l1, r1), [l2, r2), ... of 1 bits.
        Used to represent an active bucket during encoding. Use `unfold` method to get actual set of indices.
    """
    bit_ranges: List[BitRange]

    def __init__(self, bit_ranges: List[BitRange]):
        self.bit_ranges = bit_ranges

    def shift(self, shift: int) -> 'BitRangePack':
        new_bit_ranges = list()
        for bit_range in self.bit_ranges:
            new_bit_ranges.append(bit_range.shift(shift))
        return BitRangePack(new_bit_ranges)

    def unfold(self) -> Iterable[int]:
        unfolded = list()
        for bit_range in self.bit_ranges:
            unfolded.extend(bit_range.unfold())

        return unfolded


class IntSdrEncoder(Generic[ValueEncoder]):
    """
    Encodes integer values from range [0, `n_values`) as SDR w/ `total_bits` bits.
    Every single value x from the range is encoded as `value_bits` contiguous 1 bits called
        a bucket. All bits outside corresponding bucket are set to zero.

    Example:
        An encoder(n_values = 3, value_bits = 4) encodes value 1 as:
            0000 1111 0000
        Buckets are separated by empty space for clarity. Note, that there're 3 buckets of size 4.
    """
    ALL = -1

    name: str
    n_values: int
    value_bits: int
    total_bits: int
    activation_threshold: int

    def __init__(
            self, name: str, n_values: int, value_bits: int, activation_threshold: int,
            default_format: str = 'full'
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
        self.n_values = n_values
        self.value_bits = value_bits
        self.total_bits = n_values * value_bits
        self.activation_threshold = activation_threshold
        self.default_format = default_format

    def encode(self, x: int) -> BitRange:
        """
        Encodes a single value to sparse SDR in intermediate BitRange [l, r) short format.
        """
        return self._bit_bucket_range(x)

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
        decoded_values = [
            value
            for value, activation in enumerate(self.value_activations(sparse_sdr))
            if activation >= self.activation_threshold
        ]
        return decoded_values

    def has_value(self, sparse_sdr: SparseSdr, value: int) -> bool:
        """Gets whether SDR has a particular value or not."""
        value_activations = self.value_activations(sparse_sdr)
        return value_activations[value] >= self.activation_threshold

    def value_activations(self, sparse_sdr: SparseSdr) -> List[int]:
        """
        Counts number of activations in each value's bucket of bits in a given sparse SDR.
        :param sparse_sdr: sparse SDR
        :return: list of activations - i-th elem is an activation of value i.
        """
        value_activations = [0] * self.n_values
        for active_bit_index in sparse_sdr:
            value = active_bit_index // self.value_bits
            value_activations[value] += 1

        return value_activations

    def format(self, sparse_sdr: SparseSdr, format_: str = None) -> str:
        """
        Formats sparse SDR to string with one of the supported formats.

        Supported formats are: 'full' and 'short'. If None then encoder's default is used.
        """
        format_ = isnone(format_, self.default_format)
        supported_formats = {
            'full': IntSdrFormatter,
            'short': IntSdrShortFormatter,
        }
        return supported_formats[format_].format(sparse_sdr, self)

    def _bit_bucket_range(self, x: int) -> BitRange:
        """Gets BitRange [l, r) of the bit bucket corresponding to the given value `x`."""
        self._assert_acceptable_values(x)

        if x is None:
            return BitRange(0, 0)
        if x == self.ALL:
            return BitRange(0, self.total_bits)

        left = x * self.value_bits
        right = (x + 1) * self.value_bits
        return BitRange(left, right)

    def _assert_acceptable_values(self, x: int):
        assert x is None or x == self.ALL or 0 <= x < self.n_values, \
            f'Value must be in [0, {self.n_values}] or {self.ALL} or None; got {x}'

    def __str__(self):
        name, n_values, value_bits = self.name, self.n_values, self.value_bits
        return f'({name}: v{n_values} x b{value_bits})'


class IntSemanticSdrEncoder(IntSdrEncoder):
    """
    Encodes integer values from range [0, `n_values`) as SDR w/ `total_bits` bits.
    Every single value x from the range is encoded as `value_bits` contiguous 1 bits called
        a bucket. All bits outside corresponding bucket are set to zero.
    To achieve semantic similarity use step parameter. Step is a half amount of bits that make up difference
    of buckets of two adjacent integer values.


    Example:
        An encoder(n_values = 3, value_bits = 4, step=2) encodes values as:
            0: 11110000
            1: 00111100
            2: 00001111
    """

    step: int
    overlap: float

    def __init__(self, name: str, n_values: int, value_bits: int, step: int, activation_threshold: int,
                 default_format: str = 'full'):
        super(IntSemanticSdrEncoder, self).__init__(name, n_values, value_bits, activation_threshold,
                                                    default_format)
        """
        Initializes encoder.

        :param name: a name for values encoded
        :param n_values: defines a range [0, n_values) of values that can be encoded
        :param value_bits: a number of bits in a bucket used to encode single value
        :param step: a half number of bits that make up difference of buckets 
            of two adjacent integer values.
        :param activation_threshold: a number of 1 bits in a bucket, which are enough
            to activate corresponding value, i.e. this value will be decoded.
        """

        self.step = step
        self.total_bits = self.step * (self.n_values - 1) + self.value_bits
        self.overlap = (self.value_bits - self.step) / self.value_bits

    def value_activations(self, sparse_sdr: SparseSdr) -> Iterable[int]:
        """
           Counts number of activations in each value's bucket of bits in a given sparse SDR.
           :param sparse_sdr: sparse SDR
           :return: numpy array of activations - i-th elem is an activation of value i.
        """

        value_activations = np.zeros(self.n_values, dtype=np.int64)
        for active_bit_index in sparse_sdr:
            max_value = active_bit_index // self.step
            min_value = (active_bit_index - self.value_bits + 1) // self.step

            value_activations[min_value: max_value+1] += 1

        return value_activations

    def _bit_bucket_range(self, x: int) -> BitRange:
        """Gets BitRange [l, r) of the bit bucket corresponding to the given value `x`."""
        self._assert_acceptable_values(x)

        if x is None:
            return BitRange(0, 0)
        if x == self.ALL:
            return BitRange(0, self.total_bits)

        left = x * self.step
        right = left + self.value_bits
        return BitRange(left, right)

    def format(self, sparse_sdr: SparseSdr, format_: str = None) -> str:
        """
        Formats sparse SDR to string with one of the supported formats.
        """
        format_ = isnone(format_, self.default_format)
        supported_formats = {
            'full': IntSemanticSdrFormatter
        }
        return supported_formats[format_].format(sparse_sdr, self)

    def __str__(self):
        name, n_values, value_bits = self.name, self.n_values, self.value_bits
        return f'({name}: v{n_values} x b{value_bits} step: {self.step})'


class SequenceSdrEncoder(Generic[SequenceEncoder]):
    """
        Encodes sequence as SDR. Sequence should consist of numbers of types that have encoders.
        Every single value from sequence is encoded by sdr encoders that correspond to
        the sequence's values types.
        Example:
            TODO
    """
    name: str
    encoders: List[ValueEncoder]
    size: int
    total_bits: int

    def __init__(
            self, name: str, encoders: List[ValueEncoder], size: int,
            default_format: str = 'full'
    ):
        """
            Initializes encoder.

            :param name: a name for Sequence encoded
            :param encoders: encoder that will be used for encoding every value in the sequence
            :param size: size of the Sequence
            :param default_format: printing format

            TODO: topology
        """

        self.name = name
        self.encoders = encoders
        self.size = size
        self.value_bits = sum([encoder.value_bits for encoder in self.encoders])
        self.total_bits = sum([encoder.total_bits for encoder in self.encoders])
        self.activation_threshold = sum([encoder.activation_threshold for encoder in self.encoders])
        self._shifts = self._get_shifts()
        self.default_format = default_format

    def encode(self, sequence: Iterable) -> BitRangePack:
        sequence = np.array(sequence, dtype=np.int64)
        bit_ranges = list()
        for i, value, encoder in zip(count(), sequence, self.encoders):
            bit_ranges.append(encoder.encode(value).shift(self._shifts[i]))

        return BitRangePack(bit_ranges)

    def decode(self, sparse_sdr: SparseSdr):
        values = list()
        sparse_sdr = np.array(sparse_sdr, dtype=np.int64)
        for i, encoder in enumerate(self.encoders):
            values.append(encoder.decode(
                sparse_sdr[(self._shifts[i] <= sparse_sdr) &
                           (sparse_sdr < self._shifts[i + 1])]
            )
            )
        return product(*values)

    def format(self, sparse_sdr: SparseSdr, format_: str = None) -> str:
        """
        Formats sparse SDR to string with one of the supported formats.

        Supported formats are: 'full' and 'short'. If None then encoder's default is used.
        """
        format_ = isnone(format_, self.default_format)
        supported_formats = {
            'full': IntSdrFormatter,
            'short': IntSdrShortFormatter,
        }
        return supported_formats[format_].format(sparse_sdr, self)

    def _get_shifts(self):
        shifts = [0] * (1 + self.size)
        shift = 0
        for i in range(1, len(self.encoders) + 1):
            shift += self.encoders[i-1].total_bits
            shifts[i] = shift
        return shifts


class IntSdrFormatter:
    """
    Formats SDR as a dense array:
        0000 1111 0000
    """

    @staticmethod
    def format(sparse_sdr: SparseSdr, encoder: IntSdrEncoder) -> str:
        """
        :param sparse_sdr: SDR to print into string
        :param encoder: encoder for this particular kind of SDRs
        """
        dense_sdr = sparse_to_dense(sparse_sdr, encoder.total_bits)
        value_bit_buckets = (
            dense_sdr[i: i + encoder.value_bits]
            for i in range(0, encoder.total_bits, encoder.value_bits)
        )
        return ' '.join(
            ''.join(map(str, value_bits)) for value_bits in value_bit_buckets
        )


class IntSdrShortFormatter:
    """
    Formats SDR as an array of bucket activations:
        + - - . - +
    """
    format_chars = [
        '-',  # empty: zero 1 bits
        '.',  # partial:  0 < x < value_bits 1 bits
        '+',  # full value: value_bits 1 bits
    ]

    @classmethod
    def format(cls, sparse_sdr: SparseSdr, encoder: IntSdrEncoder) -> str:
        """
        :param sparse_sdr: SDR to print into string
        :param encoder: encoder for this particular kind of SDRs
        """
        bucket_activations = encoder.value_activations(sparse_sdr)

        encoded_sdr_for_printing = [
            int(value_activation > 0) + int(value_activation == encoder.value_bits)
            for value_activation in bucket_activations
        ]

        return ' '.join(cls.format_chars[x] for x in encoded_sdr_for_printing)


class IntSemanticSdrFormatter:
    """
        Formats SDR as a dense array:
            000011110000
    """

    @staticmethod
    def format(sparse_sdr: SparseSdr, encoder: IntSdrEncoder) -> str:
        """
        :param sparse_sdr: SDR to print into string
        :param encoder: encoder for this particular kind of SDRs
        """
        dense_sdr = sparse_to_dense(sparse_sdr, encoder.total_bits)
        value_bit_buckets = (
            dense_sdr[i: i + encoder.value_bits]
            for i in range(0, encoder.total_bits, encoder.value_bits)
        )
        return ''.join(
            ''.join(map(str, value_bits)) for value_bits in value_bit_buckets
        )
