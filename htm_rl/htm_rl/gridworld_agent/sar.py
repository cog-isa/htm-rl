from typing import List, TypeVar

from htm_rl.representations.sar import Sar as GenericSar
from htm_rl.utils import isnone

T = TypeVar('T')
Superposition = List[T]
Matrix = List[List[T]]

StateElem = int
State = Matrix[StateElem]
StateElemSuperposition = Superposition[StateElem]
StateSuperposition = Matrix[StateElemSuperposition]

ActionReward = int
ActionRewardSuperposition = Superposition[ActionReward]

Sar = GenericSar[State, ActionReward]
SarSuperposition = GenericSar[StateSuperposition, ActionRewardSuperposition]


def str_from_sar_superposition(sar: SarSuperposition) -> str:
    return _SarSuperpositionFormatter.format(sar)


def sar_superposition_has_reward(sar: SarSuperposition) -> bool:
    return sar.reward is not None and 1 in sar.reward


class _SarSuperpositionFormatter:
    state_chars = ['X', '-', '#', '^']
    action_chars = ['<', '>', '^']
    reward_chars = ['.', '+', '-']

    @classmethod
    def format(cls, sar: SarSuperposition) -> str:
        return ' '.join([
            cls._str_from_state_superposition(sar.state),
            cls._str_from_superposition(sar.action, cls.action_chars),
            cls._str_from_superposition(sar.reward, cls.reward_chars),
        ])

    @staticmethod
    def _str_from_superposition(x: List[int], mapping: List[str], fixed_len: int = None) -> str:
        n_x = len(isnone(x, []))
        n = max(n_x, isnone(fixed_len, len(mapping)))
        d = n - n_x
        return ''.join(
            mapping[x[i - d]] if d <= i else ' '
            for i in range(n_x + d)
        )

    @classmethod
    def _str_from_state_superposition(cls, state: StateSuperposition) -> str:
        def to_str(x: StateElemSuperposition) -> str:
            return cls._str_from_superposition(x, cls.state_chars)

        return '\n'.join(
            '|'.join(to_str(col) for col in row)
            for row in state
        )
