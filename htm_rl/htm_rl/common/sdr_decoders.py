import numpy as np
from htm.bindings.sdr import SDR
from typing import Type, Sequence, Union


class Decoder:
    def __init__(self, default_value):
        self.default_value = default_value

    def decode(self, pattern: Sequence):
        return self.default_value


class IntBucketDecoder(Decoder):
    def __init__(self, n_values, bucket_size, bucket_step=None, default_value: Union[str, int] = 'random', seed=None):
        super(IntBucketDecoder, self).__init__(default_value)

        self._rng = np.random.default_rng(seed)

        if bucket_step is not None:
            raise NotImplemented

        self.n_values = n_values
        self.bucket_size = bucket_size
        self.bucket_step = bucket_step

    def decode(self, pattern: Sequence):
        if len(pattern) > 0:
            buckets, counts = np.unique(
                np.array(pattern)//self.bucket_size,
                return_counts=True)
            buckets = buckets[counts == counts.max()]
            if buckets.size > 1:
                value = self._rng.choice(buckets)
            else:
                value = buckets[0]
        else:
            if self.default_value == 'random':
                value = self._rng.integers(self.n_values)
            else:
                value = self.default_value
        return int(value)


class DecoderStack:
    def __init__(self):
        self.decoders = list()

    def add_decoder(self, decoder: Type[Decoder], bit_range: (int, int)):
        self.decoders.append((decoder, bit_range))

    def decode(self, pattern: np.ndarray):
        values = list()
        for decoder, bit_range in self.decoders:
            mask = (pattern > bit_range[0]) & (pattern < bit_range[1])
            active_bits = pattern[mask] - bit_range[0]
            value = decoder.decode(active_bits)
            values.append(value)
        return values
