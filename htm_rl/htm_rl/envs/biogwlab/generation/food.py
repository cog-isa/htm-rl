from itertools import product
from typing import Tuple

import numpy as np

from htm_rl.common.utils import isnone


class FoodGenerator:
    BEANS_DISTRIBUTION = [.3, .1, .25, .15, .1]
    RANCID_BEANS_DISTRIBUTION = [.25, .15, .1, .35, .15]
    FRUIT_DISTRIBUTION = [.15, .35, .2, .1, .2]
    RANCID_FRUIT_DISTRIBUTION = [.15, .3, .15, .15, .25]
    FOOD_TYPES_DISTRIBUTION = [.37, .33, .17, .13]

    shape: Tuple[int, int]
    n_types: int
    n_items: int

    def __init__(self, shape, n_types, n_items):
        self.shape = shape

        self.n_types = n_types

        n_cells = shape[0] * shape[1]
        self.n_items = isnone(n_items, max(int((n_cells - 2) ** .5), self.n_types))

        food_types = np.arange(self.n_types)
        self.food_distribution = np.array([
            self.BEANS_DISTRIBUTION,
            self.RANCID_BEANS_DISTRIBUTION,
            self.FRUIT_DISTRIBUTION,
            self.RANCID_FRUIT_DISTRIBUTION
        ])
        self.food_distribution = self.food_distribution[food_types]

        self.food_types_distribution = np.array(self.FOOD_TYPES_DISTRIBUTION)[food_types]
        self.food_types_distribution /= self.food_types_distribution.sum()

    def generate(self, areas_map, obstacle_mask, seed):
        rng = np.random.default_rng(seed=seed)
        shape = areas_map.shape
        n_cells = shape[0] * shape[1]

        foods = rng.choice(
            self.n_types,
            p=self.food_types_distribution,
            size=self.n_items
        )
        # print(f'Food: {np.unique(foods, return_counts=True)[1]}')

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