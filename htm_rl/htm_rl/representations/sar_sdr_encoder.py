from typing import Tuple, List

import numpy as np

from representations.int_sdr_encoder import IntSdrEncoder
from representations.sar import Sar
from representations.sdr import DenseSdr, SparseSdr
from utils import sparse_to_dense


# do not refactor yet; split into different files, refactor class names,
# then start analysing problems top-down


class SarSdrEncoder:
    encoders: Tuple[IntSdrEncoder, IntSdrEncoder, IntSdrEncoder]
    total_bits: int
    value_bits: int

    def __init__(self, encoders: Tuple[IntSdrEncoder, IntSdrEncoder, IntSdrEncoder]):
        assert len(encoders) == 3

        self.encoders = encoders
        self.total_bits = sum(e.total_bits for e in encoders)
        self.value_bits = sum(e.value_bits for e in encoders)

    def encode_sparse(self, sar: Sar) -> SparseSdr:
        assert len(sar) == len(self.encoders)
        base_shift = 0
        encoded_indices = []
        for x, encoder in zip(sar, self.encoders):
            indices = encoder.encode_sparse(x)
            indices = self._apply_shift(indices, base_shift)

            encoded_indices.extend(indices)
            base_shift += encoder.total_bits
        return encoded_indices

    def decode_dense(self, arr: np.ndarray) -> List[List[int]]:
        base_shift = 0
        decoded_values = []
        for encoder in self.encoders:
            values = encoder.decode_dense(arr[base_shift: base_shift + encoder.total_bits])
            decoded_values.append(values)
            base_shift += encoder.total_bits

        return decoded_values

    def decode_sparse(self, indices: List[int]) -> List[List[int]]:
        return self.decode_dense(sparse_to_dense(indices, self.total_bits))

    def str_from_dense(self, arr: np.ndarray) -> str:
        assert arr.ndim == 1 and arr.shape[0] == self.total_bits, \
            f'Array shape mismatch. Expected ({self.total_bits},); got {arr.shape}'
        base_shift = 0
        substrings = []
        for encoder in self.encoders:
            substrings.append(
                encoder.str_from_dense(arr[base_shift: base_shift + encoder.total_bits])
            )
            base_shift += encoder.total_bits
        return ' '.join(substrings)

    @staticmethod
    def _apply_shift(indices: List[int], shift: int):
        return [i + shift for i in indices]

    def encode_dense(self, sar: Sar, out_result: DenseSdr = None) -> DenseSdr:
        if out_result is None:
            out_result = np.zeros(self.total_bits, dtype=np.int8)

        result = out_result
        base_shift = 0
        for x, encoder in zip(sar, self.encoders):
            if x is not None:
                l, r = base_shift, base_shift + encoder.total_bits
                result[l:r] = 0
                encoder.encode_dense(x, result[l:r])
            base_shift += encoder.total_bits
        return result

    def encode_dense_state_all_actions(self, state: int, reward: int = 0) -> np.ndarray:
        encoded_vector = np.zeros(self.total_bits, dtype=np.int8)
        state_encoder, action_encoder, reward_encoder = self.encoders

        state_encoder.encode_dense(state, encoded_vector[0: state_encoder.total_bits])

        base_shift = state_encoder.total_bits
        encoded_vector[base_shift: base_shift + action_encoder.total_bits] = 1

        base_shift += action_encoder.total_bits
        reward_encoder.encode_dense(reward, encoded_vector[base_shift: base_shift + reward_encoder.total_bits])
        return encoded_vector
