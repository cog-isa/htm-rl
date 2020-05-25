from htm_rl.representations.sar import BaseSar, Superposition

Sar = BaseSar[int, int, int]
SarSuperposition = BaseSar[Superposition, Superposition, Superposition]


def str_from_sar_superposition(sar_superposition: SarSuperposition) -> str:
    return ' '.join(
        ''.join(map(str, superposition))
        for superposition in sar_superposition
    )


def sar_superposition_has_reward(sar: SarSuperposition) -> bool:
    return sar.reward is not None and 1 in sar.reward

