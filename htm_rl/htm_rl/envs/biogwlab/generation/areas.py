from itertools import product
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

from htm_rl.common.sdr_encoders import IntArrayEncoder


class AreasGenerator:
    shape: Tuple[int, int]
    n_types: int

    def __init__(self, shape, n_types):
        self.shape = shape
        self.n_types = n_types

    def generate(self, seed):
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


class Areas:
    shape: Tuple[int, int]
    view_shape: Tuple[int, int]

    map: np.ndarray
    n_types: int

    _generator: AreasGenerator
    _encoder: IntArrayEncoder

    def __init__(self, shape, n_types, **generator):
        self.shape = shape
        self.n_types = n_types
        self._generator = AreasGenerator(shape=shape, n_types=n_types, **generator)

    def generate(self, seed):
        if self.n_types > 1:
            self.map = self._generator.generate(seed)
        else:
            self.map = np.zeros(self.shape, dtype=np.int)

    def set_renderer(self, view_shape):
        self.view_shape = view_shape
        self._encoder = IntArrayEncoder(shape=view_shape, n_types=self.n_types)
        return self._encoder

    def render(self, view_clip=None):
        if self.n_types == 1:
            return None

        if view_clip is not None:
            view_indices, abs_indices = view_clip

            area_map = np.zeros(self.view_shape, dtype=np.int).flatten()
            area_map[view_indices] = self.map.flatten()[abs_indices]
        else:
            area_map = self.map

        return self._encoder.encode(area_map)
