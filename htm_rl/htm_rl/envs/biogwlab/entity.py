from typing import Tuple

import numpy as np

from htm_rl.common.utils import isnone


class Entity:
    shape: Tuple[int, int]
    mask: np.ndarray
    map: np.ndarray
    n_types: int

    def __init__(self, shape=None, n_types=1, env=None):
        shape = isnone(shape, env.shape)
        self.shape = shape
        self.n_types = n_types

    def set(self, mask: np.ndarray, map: np.ndarray):
        self.mask = mask
        self.map = map