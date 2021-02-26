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

    def __init__(self, shape_xy: Tuple[int, int], seed: int, init=True):
        # convert from x,y to i,j
        width, height = shape_xy
        self.shape = (height, width)

        self.n_cells = height * width
        self.seed = seed

        if init:
            self.obstacle_mask = np.zeros(shape_xy, dtype=np.bool)
            self.food_mask = np.zeros(shape_xy, dtype=np.bool)
            self.agent_position = (0, 0)

    def make_copy(self):
        env = EnvironmentState(shape_xy=self.shape, seed=self.seed, init=False)
        env.obstacle_mask = self.obstacle_mask
        env.food_mask = self.food_mask.copy()
        env.n_foods = self.n_foods
        env.agent_position = self.agent_position
        return env


class BioGwLabEnvState:
    directions = [
        # (i, j) ~ (y, x); real - counter-clockwise, (i,j)-based - clockwise
        (0, 1, 'right'), (1, 0, 'down'), (0, -1, 'left'), (-1, 0, 'up')
    ]

    size: int
    seed: int
    n_types_area: int
    n_types_obstacle: int
    n_types_food: int
    n_scent_channels: int
    areas_map: np.ndarray
    obstacle_mask: np.ndarray
    obstacle_map: np.ndarray
    food_items: List[Tuple[int, int, int]]
    food_map: np.ndarray
    food_mask: np.ndarray
    food_scents: np.ndarray
    food_scent: np.ndarray
    scent: np.ndarray

    agent_initial_position: Optional[Tuple[int, int]]
    agent_initial_direction: Optional[int]
    agent_position: Optional[Tuple[int, int]]
    agent_direction: Optional[int]
    n_rewarding_foods: int

    def __init__(
            self, size: int, seed: int, n_area_types: int,
            n_wall_colors: int, n_food_types: int,
            n_scent_channels: int, obstacle_mask: np.ndarray,
            areas_map: np.ndarray, wall_colors: np.ndarray,
            food_items: List[Tuple[int, int, int]], food_map: np.ndarray,
            food_mask: np.ndarray, food_scents: np.ndarray
    ):
        self.size = size
        self.seed = seed
        self.n_types_area = n_area_types
        self.n_types_obstacle = n_wall_colors
        self.n_types_food = n_food_types
        self.n_scent_channels = n_scent_channels
        self.obstacle_mask = obstacle_mask
        self.areas_map = areas_map
        self.obstacle_map = wall_colors
        self.food_items = food_items
        self.food_map = food_map
        self.food_mask = food_mask
        self.n_rewarding_foods = 0
        self.food_scents = food_scents
        # self.food_scent = self.get_food_scent(food_scents)

        self.agent_initial_position = None
        self.agent_initial_direction = None
        self.agent_position = None
        self.agent_direction = None

    def get_food_scent(self, food_scents):
        food_scent = food_scents.sum(axis=-1)
        if self.n_rewarding_foods > 0:
            normalize_factors = (
                food_scent
                    .reshape((-1, food_scent._shape_xy[-1]))
                    .sum(axis=0)
                    .reshape((1, 1, -1))
            )
            food_scent /= normalize_factors + 1e-5
        return food_scent
