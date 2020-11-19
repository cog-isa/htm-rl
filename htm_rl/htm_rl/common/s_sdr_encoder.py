from htm_rl.common.sdr import SparseSdr
import numpy as np
from typing import Tuple
from htm_rl.envs.gridworld_pomdp import legend


class StateSDREncoder:
    """
        TODO
    """
    ALL = -1

    name: str
    n_values: int
    total_bits: int
    activation_threshold: int

    def __init__(
            self, name: str,
            n_values: int, shape: Tuple[int, int],
            threshold: float,
            default_format: str = 'full'
    ):
        """
            TODO
        """
        self.name = name
        self.n_values = n_values
        self.shape = shape
        self.total_bits = n_values * shape[0] * shape[1]
        self.default_format = default_format
        self.value_bits = shape[0] * shape[1]
        self.activation_threshold = int(threshold * self.value_bits)

    def encode(self, x: np.array) -> SparseSdr:
        """
        Encodes a state to sparse SDR.
        """
        sdr = list()
        shift = x.size
        for i in range(self.n_values):
            sdr.extend(np.flatnonzero(x == i) + i * shift)
        return sdr

    def decode(self, sparse_sdr: SparseSdr):
        return None

    def format(self, sparse_sdr: SparseSdr, format_: str = None) -> str:
        """TODO"""
        size = self.shape[0] * self.shape[1]
        res = ['0' if not ((i + 1) % self.shape[0] == 0) else '0\n' for i in range(size)]
        for x in sparse_sdr:
            number = x // size
            coord = x % size
            if (coord+1) % self.shape[0]:
                postfix = '\n'
            else:
                postfix = ''
            res[coord] = legend['map'][number] + postfix
        return ''.join(res)

    def __str__(self):
        name, n_values, shape = self.name, self.n_values, self.shape
        return f'({name}: v{n_values} x b{shape})'
