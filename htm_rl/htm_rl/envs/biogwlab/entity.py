from typing import Tuple, Optional

import numpy as np

from htm_rl.common.utils import isnone


class Entity:
    shape: Tuple[int, int]
    mask: Optional[np.ndarray]
    map: Optional[np.ndarray]
    n_types: int

    def __init__(self, shape=None, n_types=1, env=None):
        shape = isnone(shape, env.shape)
        self.shape = shape
        self.n_types = n_types
        self.mask = None
        self.map = None

    def set(self, mask: np.ndarray = None, map_: np.ndarray = None):
        self.mask = mask
        self.map = map_