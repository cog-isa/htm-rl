from typing import List, TypeVar

from htm_rl.representations.sar import Sar as GenericSar

T = TypeVar('T')
Superposition = List[T]

State = int
StateSuperposition = Superposition[State]

ActionReward = int
ActionRewardSuperposition = Superposition[ActionReward]

Sar = GenericSar[State, ActionReward]
SarSuperposition = GenericSar[StateSuperposition, ActionRewardSuperposition]


def str_from_sar_superposition(sar_superposition: SarSuperposition) -> str:
    return ' '.join(
        ''.join(map(str, superposition))
        for superposition in sar_superposition
    )


def sar_superposition_has_reward(sar: SarSuperposition) -> bool:
    return sar.reward is not None and 1 in sar.reward

