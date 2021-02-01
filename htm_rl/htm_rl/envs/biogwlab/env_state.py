from typing import Tuple, List, Optional

import numpy as np


class BioGwLabEnvState:
    directions = [(1, 0, 'right'), (0, -1, 'down'), (-1, 0, 'left'), (0, 1, 'up')]

    size: int
    seed: int
    n_area_types: int
    n_wall_colors: int
    n_food_types: int
    n_scent_channels: int
    obstacle_mask: np.ndarray
    areas_map: np.ndarray
    wall_colors: np.ndarray
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

    def __init__(
            self, size: int, seed: int, n_area_types: int,
            n_wall_colors: int, n_food_types: int,
            n_scent_channels: int, obstacle_mask: np.ndarray,
            areas_map: np.ndarray, wall_colors: np.ndarray,
            food_items: List[Tuple[int, int, int]], food_mask: np.ndarray,
            food_map: np.ndarray, food_scents: np.ndarray
    ):
        self.size = size
        self.seed = seed
        self.n_area_types = n_area_types
        self.n_wall_colors = n_wall_colors
        self.n_food_types = n_food_types
        self.n_scent_channels = n_scent_channels
        self.obstacle_mask = obstacle_mask
        self.areas_map = areas_map
        self.wall_colors = wall_colors
        self.food_items = food_items
        self.food_map = food_map
        self.food_mask = food_mask
        self.food_scents = food_scents
        self.food_scent = self.get_food_scent(food_scents)

        self.agent_initial_position = None
        self.agent_initial_direction = None
        self.agent_position = None
        self.agent_direction = None

    def get_food_scent(self, food_scents):
        food_scent = food_scents.sum(axis=-1)
        normalize_factors = (
            food_scent
                .reshape((-1, food_scent.shape[-1]))
                .sum(axis=0)
                .reshape((1, 1, -1))
        )
        food_scent /= normalize_factors
        return food_scent

