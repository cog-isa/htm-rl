from typing import List, Set

import numpy as np

from htm_rl.common.plot_utils import plot_grid_images
from htm_rl.envs.biogwlab.environment import Environment
from htm_rl.envs.biogwlab.generation.food import FoodPositionsGenerator
from htm_rl.envs.biogwlab.module import Entity, EntityType
from htm_rl.envs.biogwlab.view_clipper import ViewClip


def add_food(env, types=None, **food):
    areas = env.get_module('areas')
    obstacles = env.get_module('obstacles')

    if types is None:
        food_types = {'beans': food}
        food = dict()
    else:
        food_types = types

    return Food(
        types=food_types, env=env,
        areas=areas, obstacles=obstacles,
        **food
    )


class Food(Entity):
    family = 'food'
    type = EntityType.Consumable

    reward: float
    positions_fl: Set[int]

    generator: FoodPositionsGenerator
    env: Environment

    def __init__(
            self, env: Environment, reward: float, n_items: int = 1,
            area_weights: List[float] = None,
            **entity
    ):
        super(Food, self).__init__(**entity)

        self.reward = reward
        self.generator = FoodPositionsGenerator(
            shape=env.shape, n_items=n_items, area_weights=area_weights
        )
        self.env = env

    def generate(self, seeds):
        # we should not take this entity into account
        # in aggregated masks during generation
        self.initialized = False

        seed = seeds['food']
        empty_mask = ~self.env.aggregated_mask[EntityType.Obstacle]
        areas = self.env.entity_slices[EntityType.Area]

        area_masks = []
        for area in areas:
            mask = np.zeros_like(empty_mask)
            area.append_mask(mask)
            area_masks.append(mask)

        # TODO: both flatten and non-flatten
        self.positions_fl = set(self.generator.generate(
            empty_mask=empty_mask, area_masks=area_masks, seed=seed
        ))
        self.initialized = True

    def collect(self, position, view_direction):
        reward, success = 0, False
        position_fl = self._flatten_position(position)

        if position_fl in self.positions_fl:
            self.positions_fl.remove(position_fl)
            success = True
            reward = self.reward

        return reward, success

    def render(self, view_clip: ViewClip = None):
        if view_clip is None:
            return np.array(list(self.positions_fl)), self.env.shape[0] * self.env.shape[1]

        indices = []
        for abs_ind, view_ind in zip(view_clip.abs_indices, view_clip.view_indices):
            if abs_ind in self.positions_fl:
                indices.append(view_ind)

        return np.array(indices), view_clip.shape[0] * view_clip.shape[1]

    def append_mask(self, mask: np.ndarray):
        if not self.initialized:
            return

        for position_fl in self.positions_fl:
            pos = self._unflatten_position(position_fl)
            mask[pos] = 1

    def append_position(self, exist: bool, position):
        return exist or (
            self.initialized
            and self._flatten_position(position) in self.positions_fl
        )

    def _flatten_position(self, position):
        return position[0] * self.env.shape[1] + position[1]

    def _unflatten_position(self, position_fl):
        return divmod(position_fl, self.env.shape[1])
