from typing import Tuple, TypeVar, Generic

from htm_rl.representations.int_sdr_encoder import IntSdrEncoder
from htm_rl.representations.sar import SarRelatedComposition
from htm_rl.representations.sdr import SparseSdr

TSar = TypeVar('TSar')
TSarSuperposition = TypeVar('TSarSuperposition')
TStateEncoder = TypeVar('TStateEncoder')

SarShifts = SarRelatedComposition[int, int, int]
SarSplitSdr = SarRelatedComposition[SparseSdr, SparseSdr, SparseSdr]


class SarSdrEncoder(Generic[TSar, TSarSuperposition, TStateEncoder]):
    value_bits: int
    total_bits: int
    activation_threshold: int

    _encoders: SarRelatedComposition[TStateEncoder, IntSdrEncoder, IntSdrEncoder]
    _shifts: SarShifts

    def __init__(self, encoders):
        if not isinstance(encoders, SarRelatedComposition):
            encoders = SarRelatedComposition(*encoders)

        self.value_bits = sum(e.value_bits for e in encoders)
        self.total_bits = sum(e.total_bits for e in encoders)
        self.activation_threshold = sum(e.activation_threshold for e in encoders)

        self._shifts = SarSdrEncoder._get_shifts(encoders)
        self._encoders = encoders

    def encode(self, sar: TSar) -> SparseSdr:
        return [
            ind
            for x, encoder, shift in zip(sar, self._encoders, self._shifts)
            for bit_range in encoder.encode(x, shift)
            for ind in range(bit_range.l, bit_range.r)
        ]

    def decode(self, indices: SparseSdr) -> TSarSuperposition:
        split_indices = self._split_indices_between_encoders(indices)
        states, actions, rewards = (
            encoder.decode(indices_)
            for encoder, indices_ in zip(self._encoders, split_indices)
        )
        return SarRelatedComposition(states, actions, rewards)

    def get_rewarding_indices_range(self) -> Tuple[int, int]:
        bit_range, *_ = self._encoders.reward.encode(1, self._shifts.reward)
        l, r = bit_range
        return l, r

    def get_actions_indices_range(self) -> Tuple[int, int]:
        action_shift = self._shifts[1]
        action_encoder = self._encoders[1]
        return action_shift, action_shift + action_encoder.total_bits

    def _split_indices_between_encoders(self, indices: SparseSdr) -> SarSplitSdr:
        shifts = self._shifts
        split_indices = SarSplitSdr([], [], [])

        def put_into_corresponding_bucket(ind: int):
            if ind < shifts.action:
                split_indices.state.append(ind)
            elif ind < shifts.reward:
                split_indices.action.append(ind - shifts.action)
            else:
                split_indices.reward.append(ind - shifts.reward)

        for i in indices:
            put_into_corresponding_bucket(i)
        return split_indices

    def format(self, indices: SparseSdr) -> str:
        split_indices = self._split_indices_between_encoders(indices)
        return ' | '.join(
            encoder.format(indices_)
            for encoder, indices_ in zip(self._encoders, split_indices)
        )

    @staticmethod
    def _get_shifts(encoders) -> SarShifts:
        actions_shift = encoders.state.total_bits
        rewards_shift = actions_shift + encoders.action.total_bits
        return SarShifts(0, actions_shift, rewards_shift)
