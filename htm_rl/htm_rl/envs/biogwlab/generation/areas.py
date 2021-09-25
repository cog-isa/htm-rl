from itertools import product

import numpy as np


class MultiAreaMapGenerator:
    shape: tuple[int, int]
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
        area_centers = np.divmod(flatten_centers, width)
        area_types = rng.integers(self.n_types, size=n_areas)
        return area_centers, area_types

    def _debug_plot_area_centers(self, area_centers, area_types):
        from matplotlib import pyplot as plt

        size = self.shape
        area = np.zeros((size, size), dtype=np.int8)
        for i, t in enumerate(area_types):
            area[area_centers[0][i], area_centers[1][i]] = t + 1

        plt.imshow(area)
        plt.show()