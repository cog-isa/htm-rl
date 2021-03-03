from itertools import product
from typing import Tuple, List

import numpy as np

from htm_rl.common.sdr_encoders import IntArrayEncoder
from htm_rl.common.utils import isnone


class FoodGenerator:
    BEANS_DISTRIBUTION = [.3, .1, .25, .15, .1]
    RANCID_BEANS_DISTRIBUTION = [.25, .15, .1, .35, .15]
    FRUIT_DISTRIBUTION = [.15, .35, .2, .1, .2]
    RANCID_FRUIT_DISTRIBUTION = [.15, .3, .15, .15, .25]
    FOOD_TYPES_DISTRIBUTION = [.37, .33, .17, .13]

    def __init__(self, food_types):
        self.food_distribution = np.array([
            self.BEANS_DISTRIBUTION,
            self.RANCID_BEANS_DISTRIBUTION,
            self.FRUIT_DISTRIBUTION,
            self.RANCID_FRUIT_DISTRIBUTION
        ])
        self.food_distribution = self.food_distribution[food_types]

        self.food_types_distribution = np.array(self.FOOD_TYPES_DISTRIBUTION)[food_types]
        self.food_types_distribution /= self.food_types_distribution.sum()

        self.n_food_types = len(self.food_types_distribution)

    def generate(self, areas_map, obstacle_mask, seed, n_foods = None):
        rng = np.random.default_rng(seed=seed)
        shape = areas_map.shape
        n_cells = shape[0] * shape[1]
        n_foods = isnone(n_foods, max(int((n_cells - 2) ** .5), self.n_food_types))

        foods = rng.choice(
            self.n_food_types,
            p=self.food_types_distribution,
            size=n_foods
        )
        print(f'Food: {np.unique(foods, return_counts=True)[1]}')

        food_mask = np.zeros(shape, dtype=np.bool)
        food_items = []
        food_map = np.full(shape, -1, dtype=np.int)
        food_probs = np.empty(shape, dtype=np.float)
        for food_item in foods:
            food_probs.fill(0.)
            for pos in product(range(shape[0]), range(shape[1])):
                if obstacle_mask[pos] or food_mask[pos]:
                    continue
                food_probs[pos] = self.food_distribution[food_item, areas_map[pos]]

            food_probs /= food_probs.sum()
            ind = rng.choice(n_cells, p=food_probs.flatten())
            i, j = divmod(ind, shape[1])
            food_items.append((i, j, food_item))
            food_mask[i, j] = True
            food_map[i, j] = food_item
        return food_items, food_map, food_mask


class Food:
    shape: Tuple[int, int]
    view_shape: Tuple[int, int]

    mask: np.ndarray
    map: np.ndarray
    items: List[Tuple[int, int, int]]
    n_types: int
    n_items: int
    n_foods_to_find: int

    rewards: List[int]

    _generator: FoodGenerator
    _encoder: IntArrayEncoder

    def __init__(
            self, shape,
            n_types: int = None, reward: int = None,
            food_types: List[int] = None, rewards: List[float] = None,
            n_items=None, n_foods_to_find=None
    ):
        self.shape = shape

        food_types = isnone(food_types, [0])
        self.n_types = isnone(n_types, len(food_types))
        self.rewards = isnone(rewards, [reward])
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
