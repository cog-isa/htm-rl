from copy import copy
from typing import Tuple, List, Optional

import numpy as np


class EnvironmentState:
    directions = {
        # (i, j)-based coordinates [or (y, x) for the viewer]
        'right': (0, 1), 'down': (1, 0), 'left': (0, -1), 'up': (-1, 0)
    }

    seed: int
    shape: Tuple[int, int]
    n_cells: int
    obstacle_mask: np.ndarray
    food_mask: np.ndarray
    n_foods: int
    agent_position: Tuple[int, int]
    episode_step: int
    reward: float

    def __init__(self, shape_xy: Tuple[int, int], seed: int, init=True):
        # convert from x,y to i,j
        width, height = shape_xy
        self.shape = (height, width)

        self.n_cells = height * width
        self.seed = seed
        self.episode_step = 0
        self.reward = 0

        if init:
            self.obstacle_mask = np.zeros(shape_xy, dtype=np.bool)
            self.food_mask = np.zeros(shape_xy, dtype=np.bool)
            self.agent_position = (0, 0)

    def make_copy(self):
        env = copy(self)
        env.food_mask = self.food_mask.copy()
        return env
