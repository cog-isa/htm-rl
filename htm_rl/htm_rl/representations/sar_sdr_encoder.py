from typing import Tuple, List, NamedTuple

import numpy as np

from representations.int_sdr_encoder import IntSdrEncoder
from representations.sar import Sar, SarSuperposition
from representations.sdr import DenseSdr, SparseSdr

# do not refactor yet; split into different files, refactor class names,
# then start analysing problems top-down

SarSdrEncodersNT = NamedTuple('SarSdrEncodersNT', [
    ('s', IntSdrEncoder),
    ('a', IntSdrEncoder),
    ('r', IntSdrEncoder),
])


class SarSdrEncoder:
    encoders: SarSdrEncodersNT
    total_bits: int
    value_bits: int

    _shifts: Tuple[int, int, int]

    def __init__(self, encoders: SarSdrEncodersNT):
        self.encoders = encoders
        self.total_bits = sum(e.total_bits for e in encoders)
        self.value_bits = sum(e.value_bits for e in encoders)
        self._shifts = SarSdrEncoder._get_shifts(encoders)

    def encode_sparse(self, sar: Sar) -> SparseSdr:
        encoded_indices = [
            ind
            for x, encoder, shift in zip(sar, self.encoders, self._shifts)
            for ind in encoder.encode_sparse_with_shift(x, shift)
        ]
        return encoded_indices

    def decode_sparse(self, indices: SparseSdr) -> SarSuperposition:
        split_indices = self._split_indices_between_encoders(indices)
        states, actions, rewards = (
            encoder.decode_sparse(indices_)
            for encoder, indices_ in zip(self.encoders, split_indices)
        )
        return SarSuperposition(states, actions, rewards)

    def _split_indices_between_encoders(self, indices: SparseSdr) -> Tuple[SparseSdr, ...]:
        state_shift, action_shift, reward_shift = self._shifts
        state_indices, action_indices, reward_indices = [], [], []

        def put_in_corresponding_basket(i: int):
            if i < action_shift:
                state_indices.append(i)
            elif i < reward_shift:
                action_indices.append(i - action_shift)
            else:
                reward_indices.append(i - reward_shift)

        for i in indices:
            put_in_corresponding_basket(i)
        return state_indices, action_indices, reward_indices

    def str_from_sparse(self, indices: SparseSdr) -> str:
        split_indices = self._split_indices_between_encoders(indices)
        return ' '.join(
            encoder.str_from_sparse(indices_)
            for encoder, indices_ in zip(self.encoders, split_indices)
        )

    def _str_from_dense(self, arr: DenseSdr) -> str:
        assert arr.ndim == 1 and arr.shape[0] == self.total_bits, \
            f'Array shape mismatch. Expected ({self.total_bits},); got {arr.shape}'
        return ' '.join(
            encoder._str_from_dense(arr[shift: shift + encoder.total_bits])
            for encoder, shift in zip(self.encoders, self._shifts)
        )

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

    @staticmethod
    def _get_shifts(encoders: SarSdrEncodersNT) -> Tuple[int, int, int]:
        actions_shift = encoders.s.total_bits
        rewards_shift = actions_shift + encoders.a.total_bits
        return 0, actions_shift, rewards_shift


# class SarSdrEncoder:
#     encoders: Tuple[IntSdrEncoder, IntSdrEncoder, IntSdrEncoder]
#     total_bits: int
#     value_bits: int
#
#     def __init__(self, encoders: Tuple[IntSdrEncoder, IntSdrEncoder, IntSdrEncoder]):
#         assert len(encoders) == 3
#
#         self.encoders = encoders
#         self.total_bits = sum(e.total_bits for e in encoders)
#         self.value_bits = sum(e.value_bits for e in encoders)
#
#     def encode_sparse(self, sar: Sar) -> SparseSdr:
#         assert len(sar) == len(self.encoders)
#         base_shift = 0
#         encoded_indices = []
#         for x, encoder in zip(sar, self.encoders):
#             indices = encoder.encode_sparse(x)
#             indices = self._apply_shift(indices, base_shift)
#
#             encoded_indices.extend(indices)
#             base_shift += encoder.total_bits
#         return encoded_indices
#
#     def decode_dense(self, arr: np.ndarray) -> List[List[int]]:
#         base_shift = 0
#         decoded_values = []
#         for encoder in self.encoders:
#             values = encoder.decode_dense(arr[base_shift: base_shift + encoder.total_bits])
#             decoded_values.append(values)
#             base_shift += encoder.total_bits
#
#         return decoded_values
#
#     def decode_sparse(self, indices: List[int]) -> List[List[int]]:
#         return self.decode_dense(sparse_to_dense(indices, self.total_bits))
#
#     def str_from_dense(self, arr: np.ndarray) -> str:
#         assert arr.ndim == 1 and arr.shape[0] == self.total_bits, \
#             f'Array shape mismatch. Expected ({self.total_bits},); got {arr.shape}'
#         base_shift = 0
#         substrings = []
#         for encoder in self.encoders:
#             substrings.append(
#                 encoder.str_from_dense(arr[base_shift: base_shift + encoder.total_bits])
#             )
#             base_shift += encoder.total_bits
#         return ' '.join(substrings)
#
#     @staticmethod
#     def _apply_shift(indices: List[int], shift: int):
#         return [i + shift for i in indices]
#
#     def encode_dense(self, sar: Sar, out_result: DenseSdr = None) -> DenseSdr:
#         if out_result is None:
#             out_result = np.zeros(self.total_bits, dtype=np.int8)
#
#         result = out_result
#         base_shift = 0
#         for x, encoder in zip(sar, self.encoders):
#             if x is not None:
#                 l, r = base_shift, base_shift + encoder.total_bits
#                 result[l:r] = 0
#                 encoder.encode_dense(x, result[l:r])
#             base_shift += encoder.total_bits
#         return result
#
#     def encode_dense_state_all_actions(self, state: int, reward: int = 0) -> np.ndarray:
#         encoded_vector = np.zeros(self.total_bits, dtype=np.int8)
#         state_encoder, action_encoder, reward_encoder = self.encoders
#
#         state_encoder.encode_dense(state, encoded_vector[0: state_encoder.total_bits])
#
#         base_shift = state_encoder.total_bits
#         encoded_vector[base_shift: base_shift + action_encoder.total_bits] = 1
#
#         base_shift += action_encoder.total_bits
#         reward_encoder.encode_dense(reward, encoded_vector[base_shift: base_shift + reward_encoder.total_bits])
#         return encoded_vector
