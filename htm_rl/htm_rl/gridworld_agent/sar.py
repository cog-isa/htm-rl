from typing import List, TypeVar

from htm_rl.representations.sar import BaseSar, Superposition
from htm_rl.utils import isnone

T = TypeVar('T')

SuperpositionList = List[Superposition]

Sar = BaseSar[List[int], int, int]
SarSuperposition = BaseSar[SuperpositionList, Superposition, Superposition]


def str_from_sar_superposition(sar: SarSuperposition) -> str:
    return SarSuperpositionFormatter.format(sar)


def sar_superposition_has_reward(sar: SarSuperposition) -> bool:
    return sar.reward is not None and 1 in sar.reward


class SarSuperpositionFormatter:
    state_chars = ['X', '-', '#', '^']
    action_chars = ['<', '>', '^']
    reward_chars = ['.', '+', '-']

    n_rows: int
    n_cols: int

    def __init__(self, n_rows: int, n_cols: int):
        self.n_rows = n_rows
        self.n_cols = n_cols

    def format(self, sar: SarSuperposition) -> str:
        return ' '.join([
            self._str_from_state_superposition(sar.state),
            self._str_from_superposition(sar.action, self.action_chars),
            self._str_from_superposition(sar.reward, self.reward_chars),
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

    def _str_from_state_superposition(self, state: List[Superposition]) -> str:
        def to_str(x: Superposition) -> str:
            return self._str_from_superposition(x, self.state_chars)

        n_rows, n_cols = self.n_rows, self.n_cols
        return '\n'.join(
            '|'.join(to_str(state[row * n_cols + col]) for col in range(n_cols))
            for row in range(n_rows)
        )
