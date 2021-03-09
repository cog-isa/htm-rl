from typing import Tuple, List

import numpy as np

from htm_rl.common.sdr_encoders import IntArrayEncoder
from htm_rl.common.utils import isnone
from htm_rl.envs.biogwlab.food_generator import FoodGenerator


def add_food(env, types=None, **food):
    if types is None:
        food_types = {'beans': food}
    else:
        food_types = types

    # rewards = dict()

    # for food_type, food_type_config in food_types.items():
    #     rewards[food_type] = food_type_config['reward']
    #     food_type_config.pop('reward')

    areas = env.get_module('areas')
    obstacles = env.get_module('obstacles')

    return FoodGenerator(
        types=food_types, env=env,
        areas=areas, obstacles=obstacles,
    )


class Food:
    shape: Tuple[int, int]
    view_shape: Tuple[int, int]

    mask: np.ndarray
    map: np.ndarray
    items: List[Tuple[int, int, int]]
    n_types: int
    n_items: int
    n_foods_to_find: int

    rewards: np.ndarray

    _generator: FoodGenerator
    _encoder: IntArrayEncoder

    _mask_copy: np.ndarray
    _n_items_fo_find_copy: int

    def __init__(
            self, shape,
            n_types: int = None, food_types: List[int] = None,
            reward: int = None, rewards: List[float] = None,
            n_items=None, n_foods_to_find=None
    ):
        self.shape = shape

        food_types = isnone(food_types, [0])
        self.n_types = isnone(n_types, len(food_types))
        self.rewards = np.array(isnone(rewards, [reward]))
        self.n_items = n_items
        self.n_foods_to_find = n_foods_to_find

        self._generator = FoodGenerator(food_types=food_types)

    def generate(self, seed, obstacle_mask, areas_map):
        if self.n_types == 1:
            rng = np.random.default_rng(seed=seed)

            # work in flatten then reshape
            empty_positions = np.flatnonzero(~obstacle_mask)
            food_positions = rng.choice(empty_positions, size=self.n_items, replace=False)
            food_positions = np.divmod(food_positions, self.shape[1])

            self.mask = np.zeros(self.shape, dtype=np.bool)
            self.mask[food_positions] = True

            self.map = (~self.mask).astype(np.int)

            n_positive_foods = self.n_items
        else:
            self.items, self.map, self.mask = self._generator.generate(
                areas_map=areas_map,
                obstacle_mask=obstacle_mask,
                seed=seed,
                n_foods=self.n_items
            )
            self.n_items = len(self.items)
            n_positive_foods = np.sum([
                1
                for _, _, food_type in self.items
                if self.rewards[food_type] > 0
            ])

        self.n_foods_to_find = isnone(self.n_foods_to_find, (n_positive_foods - 1) // 3 + 1)

        self._mask_copy = self.mask.copy()
        self._n_items_fo_find_copy = self.n_foods_to_find

    def set_renderer(self, view_shape):
        self.view_shape = view_shape
        self._encoder = IntArrayEncoder(shape=view_shape, n_types=self.n_types)
        return self._encoder

    def render(self, view_clip=None):
        if view_clip is None:
            return self._encoder.encode(self.map, self.mask)

        view_indices, abs_indices = view_clip

        food_mask = np.ones(self.view_shape, dtype=np.bool).flatten()
        food_mask[view_indices] = self.mask.flatten()[abs_indices]

        food_map = np.zeros(self.view_shape, dtype=np.int).flatten()
        food_map[view_indices] = self.map.flatten()[abs_indices]
        return self._encoder.encode(food_map, food_mask)

    def render_rgb(self, img: np.ndarray):
        norm_rewards = 5 * self.rewards[self.map[self.mask]] / np.abs(self.rewards).max()
        norm_rewards[norm_rewards > 0] = norm_rewards[norm_rewards > 0].astype(np.int8) + 12
        norm_rewards[norm_rewards < 0] = norm_rewards[norm_rewards < 0].astype(np.int8) - 4
        print(norm_rewards)
        img[self.mask] = norm_rewards
        print(self.items)

    def reset(self, ):
        self.mask = self._mask_copy.copy()
        self.n_foods_to_find = self._n_items_fo_find_copy

