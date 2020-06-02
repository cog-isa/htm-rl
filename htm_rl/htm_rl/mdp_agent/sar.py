from htm_rl.representations.sar import BaseSar, Superposition

Sar = BaseSar[int, int, int]
SarSuperposition = BaseSar[Superposition, Superposition, Superposition]


class SarSuperpositionFormatter:
    @staticmethod
    def format(sar: SarSuperposition) -> str:
        return ' '.join(
            '.'.join(map(str, superposition))
            for superposition in sar
        )

