from typing import List, Optional, NamedTuple

Sar = NamedTuple('Sar', [
    ('state', Optional[int]),
    ('action', Optional[int]),
    ('reward', Optional[int])
])

SarSuperposition = NamedTuple('SarSuperposition', [
    ('states', Optional[List[int]]),
    ('actions', Optional[List[int]]),
    ('rewards', Optional[List[int]])
])


def str_from_sar_superposition(sar_superposition: SarSuperposition) -> str:
    return ' '.join(
        ''.join(map(str, superposition))
        for superposition in sar_superposition
    )


def sar_superposition_has_reward(sar: SarSuperposition) -> bool:
    return sar.rewards is not None and 1 in sar.rewards
