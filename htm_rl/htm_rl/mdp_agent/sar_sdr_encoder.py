from typing import Tuple, List, NamedTuple

from htm_rl.mdp_agent.sar import Sar, SarSuperposition
from htm_rl.representations.int_sdr_encoder import IntSdrEncoder
from htm_rl.representations.sdr import SparseSdr

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
            for ind in encoder.encode(x, shift)
        ]
        return encoded_indices

    def decode_sparse(self, indices: SparseSdr) -> SarSuperposition:
        split_indices = self._split_indices_between_encoders(indices)
        states, actions, rewards = (
            encoder.decode(indices_)
            for encoder, indices_ in zip(self.encoders, split_indices)
        )
        return SarSuperposition(states, actions, rewards)

    def get_rewarding_indices_range(self) -> Tuple[int, int]:
        reward_shift = self._shifts[2]
        reward_encoder = self.encoders[2]
        return reward_shift, reward_shift + reward_encoder.total_bits

    def get_actions_indices_range(self) -> Tuple[int, int]:
        action_shift = self._shifts[1]
        action_encoder = self.encoders[1]
        return action_shift, action_shift + action_encoder.total_bits

    def _split_indices_between_encoders(self, indices: SparseSdr) -> Tuple[SparseSdr, ...]:
        state_shift, action_shift, reward_shift = self._shifts
        state_indices, action_indices, reward_indices = [], [], []

        def put_into_corresponding_bucket(ind: int):
            if ind < action_shift:
                state_indices.append(ind)
            elif ind < reward_shift:
                action_indices.append(ind - action_shift)
            else:
                reward_indices.append(ind - reward_shift)

        for i in indices:
            put_into_corresponding_bucket(i)
        return state_indices, action_indices, reward_indices

    def str_from_sparse(self, indices: SparseSdr) -> str:
        split_indices = self._split_indices_between_encoders(indices)
        return ' | '.join(
            encoder.format(indices_)
            for encoder, indices_ in zip(self.encoders, split_indices)
        )

    @staticmethod
    def _get_shifts(encoders: SarSdrEncodersNT) -> Tuple[int, int, int]:
        actions_shift = encoders.s.total_bits
        rewards_shift = actions_shift + encoders.a.total_bits
        return 0, actions_shift, rewards_shift
