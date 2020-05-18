from typing import Tuple, List, Union

SAR = Tuple[
    Union[int, None], Union[int, None], Union[int, None]
]
SAR_Superposition = Tuple[
    Union[List[int], None], Union[List[int], None], Union[List[int], None]
]


def str_from_sar_superposition(sar_superposition: SAR_Superposition) -> str:
    return ' '.join(
        ''.join(map(str, superposition))
            for superposition in sar_superposition
    )


def sar_superposition_has_reward(sar_superposition: SAR_Superposition) -> bool:
    reward_superposition = sar_superposition[2]
    return reward_superposition is not None and 1 in reward_superposition
