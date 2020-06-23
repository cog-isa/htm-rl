from dataclasses import dataclass, astuple
from typing import Iterable, List

from htm_rl.common.base_sar import Superposition
from htm_rl.common.sdr import SparseSdr, sparse_to_dense
from htm_rl.common.utils import isnone


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


class IntSdrEncoder:
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
        '-',    # empty: zero 1 bits
        '.',    # partial:  0 < x < value_bits 1 bits
        '+',    # full value: value_bits 1 bits
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
