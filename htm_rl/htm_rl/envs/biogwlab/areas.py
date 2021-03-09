from typing import Tuple

import numpy as np

from htm_rl.common.sdr_encoders import IntArrayEncoder
from htm_rl.common.utils import isnone
from htm_rl.envs.biogwlab.areas_generator import AreasGenerator
from htm_rl.envs.biogwlab.entity import Entity


def add_areas(**areas):
    return AreasGenerator(**areas)


class AreasRenderer:
    areas: Entity
    view_shape: Tuple[int, int]

    _encoder: IntArrayEncoder

    def __init__(self, view_shape):
        self.view_shape = view_shape

    def set_renderer(self, view_shape):
        self.view_shape = view_shape
        self._encoder = IntArrayEncoder(shape=view_shape, n_types=self.areas.n_types)
        return self._encoder

    def render(self, view_clip=None):
        if view_clip is not None:
            view_indices, abs_indices = view_clip

            area_map = np.zeros(self.view_shape, dtype=np.int).flatten()
            area_map[view_indices] = self.areas.map.flatten()[abs_indices]
        else:
            area_map = self.areas.map

        return self._encoder.encode(area_map)

    def render_rgb(self, img: np.ndarray):
        img[:] = self.areas.map


class Areas:
    shape: Tuple[int, int]
    view_shape: Tuple[int, int]

    map: np.ndarray
    n_types: int

    _generator: AreasGenerator
    _encoder: IntArrayEncoder

    def __init__(self, shape, n_types=None, **generator):
        self.shape = shape
        self.n_types = isnone(n_types, 1)
        self._generator = AreasGenerator(shape=shape, n_types=n_types, **generator)

    def generate(self, seed):
        if self.n_types > 1:
            self.map = self._generator.generate(seed)
        else:
            self.map = np.zeros(self.shape, dtype=np.int)

    def set_renderer(self, view_shape):
        if self.n_types == 1:
            return None

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

    def render_rgb(self, img: np.ndarray):
        img[:] = self.map
