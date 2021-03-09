from itertools import product
from typing import Tuple, Optional

import numpy as np
from matplotlib import pyplot as plt

from htm_rl.envs.biogwlab.entity import Entity


class AreasGenerator(Entity):
    entity = 'areas'

    _generator: '_AreasGenerator'
    _last_seed: Optional[int]

    def __init__(self, **areas):
        super(AreasGenerator, self).__init__(**areas)
        self._generator = _AreasGenerator(shape=self.shape, n_types=self.n_types)

    def generate(self, seed):
        if self._last_seed is not None and self._last_seed == seed:
            return

        self.set(
            mask=np.ndarray(self.shape, dtype=np.bool),
            map=self._generator.generate(seed)
        )


class _AreasGenerator:
    shape: Tuple[int, int]
    n_types: int

    def __init__(self, shape, n_types):
        self.shape = shape
        self.n_types = n_types

    def generate(self, seed):
        if self.n_types == 1:
            return np.zeros(self.shape, dtype=np.int)

        rng = np.random.default_rng(seed=seed)
        area_centers, area_types = self._spawn_area_centers(rng)
        areas_map = self._spawn_areas(area_centers, area_types)
        return areas_map

    def _spawn_areas(self, area_centers, area_types):
        height, width = self.shape
        n_centers = len(area_types)

        areas_map = np.empty(self.shape, dtype=np.int8)
        for i, j in product(range(height), range(width)):
            distances = [
                self._dist(i, j, area_centers[0][k], area_centers[1][k])
                for k in range(n_centers)
            ]
            areas_map[i, j] = area_types[np.argmin(distances)]
        return areas_map

    @staticmethod
    def _dist(i, j, _i, _j):
        return (i - _i)**2 + (j - _j)**2

    def _spawn_area_centers(self, rng):
        height, width = self.shape
        n_cells = height * width

        n_areas = rng.integers(
            max(int(n_cells/3), self.n_types),
            max(int(1.7 * n_cells), self.n_types)
        )
        flatten_centers = rng.integers(n_cells, size=n_areas)
        area_centers = np.divmod(flatten_centers, n_cells)
        area_types = rng.integers(self.n_types, size=n_areas)
        return area_centers, area_types

    def _plot_area_centers(self, area_centers, area_types):
        size = self.shape
        area = np.zeros((size, size), dtype=np.int8)
        for i, t in enumerate(area_types):
            area[area_centers[0][i], area_centers[1][i]] = t + 1

        plt.imshow(area)
        plt.show()