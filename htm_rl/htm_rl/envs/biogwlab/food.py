from typing import Optional

import numpy as np

from htm_rl.envs.biogwlab.entity import Entity
from htm_rl.envs.biogwlab.generation.food import FoodGenerator


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
    entity = 'food'

    _areas: Entity
    _obstacles: Entity

    _rewards: np.ndarray

    _generator: FoodGenerator
    _initial_mask: np.ndarray
    _last_seed: Optional[int]

    def __init__(self, types, areas, obstacles, n_items=None, **food):
        n_types = len(types)
        super(Food, self).__init__(n_types=n_types, **food)

        self._areas = areas
        self._obstacles = obstacles

        if n_items is None:
            n_items = sum(food_type['n_items'] for food_type in types.values())

        self._rewards = np.array([food_type['reward'] for food_type in types.values()], dtype=np.float)

        self._generator = FoodGenerator(
            shape=self.shape, n_types=self.n_types, n_items=n_items
        )
        self._last_seed = None

    def generate(self, seed):
        if self._last_seed == seed:
            self.mask = self._initial_mask.copy()
            return

        items, food_map, food_mask = self._generator.generate(
            areas_map=self._areas.map,
            obstacle_mask=self._obstacles.mask,
            seed=seed
        )

        self.set(mask=food_mask, map=food_map)
        self._initial_mask = food_mask.copy()
        self._last_seed = seed

    def collect(self, position, view_direction):
        reward, success = 0, False
        if self.mask[position]:
            self.mask[position] = False
            success = True
            if self.n_types == 1:
                reward = self._rewards[0]
            else:
                reward = self._rewards[self.map[position]]

        return reward, success
