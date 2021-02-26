from itertools import product

import matplotlib.pyplot as plt
import numpy as np

from htm_rl.common.utils import trace


class AreasGenerator:
    size: int
    n_area_types: int
    verbosity: int

    def __init__(self, size, n_area_types, verbosity):
        self.size = size
        self.n_area_types = n_area_types
        self.verbosity = verbosity

    def generate(self, seed):
        rnd = np.random.default_rng(seed=seed)
        area_centers, area_types = self._spawn_area_centers(rnd)
        trace(self.verbosity, 3, area_types)
        # self._plot_area_centers(area_centers, area_types)
        areas_map = self._spawn_areas(area_centers, area_types)

        trace_image(self.verbosity, 3, areas_map)
        return areas_map, self.n_area_types

    def _spawn_areas(self, area_centers, area_types):
        size = self.size
        n_centers = len(area_types)

        areas_map = np.empty((size, size), dtype=np.int8)
        for i, j in product(range(size), range(size)):
            distances = [
                self._dist(i, j, area_centers[0][k], area_centers[1][k])
                for k in range(n_centers)
            ]
            areas_map[i, j] = area_types[np.argmin(distances)]
        return areas_map

    @staticmethod
    def _dist(i, j, _i, _j):
        return (i - _i)**2 + (j - _j)**2

    def _spawn_area_centers(self, rnd):
        size = self.size
        n_areas = rnd.integers(
            max(size//3, self.n_area_types),
            max(int(1.7 * self.size), self.n_area_types)
        )
        trace(self.verbosity, 2, f'{n_areas} areas of {self.n_area_types} types are spawned')

        flatten_centers = rnd.integers(size * size, size=n_areas)
        area_centers = np.divmod(flatten_centers, size)
        area_types = rnd.integers(self.n_area_types, size=n_areas)
        return area_centers, area_types

    def _plot_area_centers(self, area_centers, area_types):
        size = self.size
        area = np.zeros((size, size), dtype=np.int8)
        for i, t in enumerate(area_types):
            area[area_centers[0][i], area_centers[1][i]] = t + 1

        plt.imshow(area)
        plt.show()
