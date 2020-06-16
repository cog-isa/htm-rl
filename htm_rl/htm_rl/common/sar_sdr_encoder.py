from htm_rl.common.base_sar import SarRelatedComposition, SarSuperposition
from htm_rl.common.int_sdr_encoder import BitRange
from htm_rl.common.sdr import SparseSdr

SarShifts = SarRelatedComposition[int, int, int]
SarSplitSdr = SarRelatedComposition[SparseSdr, SparseSdr, SparseSdr]


class SarSdrEncoder:
    """
    Composes 3 encoders for states, actions and reward respectively.
    As a whole encodes/decodes sar values + adds some sar-related logic like:
        - getting rewarding SDR indices
        - getting actions SDR indices
    """
    value_bits: int
    total_bits: int
    activation_threshold: int

    _encoders: SarRelatedComposition
    _shifts: SarRelatedComposition

    def __init__(self, encoders):
        if not isinstance(encoders, SarRelatedComposition):
            encoders = SarRelatedComposition(*encoders)

        self.value_bits = sum(e.value_bits for e in encoders)
        self.total_bits = sum(e.total_bits for e in encoders)
        self.activation_threshold = sum(e.activation_threshold for e in encoders)

        self._shifts = self._get_shifts(encoders)
        self._encoders = encoders

    def encode(self, sar) -> SparseSdr:
        return [
            bit_index
            for x, encoder, shift in zip(sar, self._encoders, self._shifts)
            for bit_index in encoder.encode(x).shift(shift).unfold()
        ]

    def decode(self, indices: SparseSdr) -> SarSuperposition:
        split_indices = self.split_between_encoders(indices)
        state_superposition, action_superposition, reward_superposition = (
            encoder.decode(indices_)
            for encoder, indices_ in zip(self._encoders, split_indices)
        )
        return SarRelatedComposition(state_superposition, action_superposition, reward_superposition)

    def get_rewarding_indices_range(self) -> BitRange:
        return self._encoders.reward.encode(1).shift(self._shifts.reward)

    def get_actions_indices_range(self) -> BitRange:
        action_shift = self._shifts.action
        reward_shift = self._shifts.reward
        return BitRange(action_shift, reward_shift)

    def split_between_encoders(self, indices: SparseSdr) -> SarSplitSdr:
        shifts = self._shifts
        split_indices = SarSplitSdr([], [], [])

        def put_for_corresponding_encoder(ind: int):
            if ind < shifts.action:
                split_indices.state.append(ind)
            elif ind < shifts.reward:
                split_indices.action.append(ind - shifts.action)
            else:
                split_indices.reward.append(ind - shifts.reward)

        for i in indices:
            put_for_corresponding_encoder(i)
        return split_indices

    def format(self, sparse_sdr: SparseSdr, format_: str = None) -> str:
        split_sdrs = self.split_between_encoders(sparse_sdr)
        return ' | '.join(
            encoder.format(sparse_sdr_, format_)
            for encoder, sparse_sdr_ in zip(self._encoders, split_sdrs)
        )

    @staticmethod
    def _get_shifts(encoders) -> SarRelatedComposition:
        actions_shift = encoders.state.total_bits
        rewards_shift = actions_shift + encoders.action.total_bits
        return SarRelatedComposition(0, actions_shift, rewards_shift)
