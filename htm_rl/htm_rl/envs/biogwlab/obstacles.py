from typing import Tuple

import numpy as np

from htm_rl.common.sdr_encoders import IntArrayEncoder
from htm_rl.envs.biogwlab.obstacles_generator import ObstaclesGenerator


class Obstacles:
    shape: Tuple[int, int]
    view_shape: Tuple[int, int]

    mask: np.ndarray
    map: np.ndarray

    _generator: ObstaclesGenerator
    _encoder: IntArrayEncoder

    def __init__(self, shape, **generator):
        self.shape = shape
        self._generator = ObstaclesGenerator(shape=self.shape, **generator)

    def generate(self, seed):
        self.mask = self._generator.generate(seed=seed)
        self.map = (~self.mask).astype(np.int)

    def set_renderer(self, view_shape):
        self.view_shape = view_shape
        self._encoder = IntArrayEncoder(shape=view_shape, n_types=1)
        return self._encoder

    def render(self, view_clip=None):
        if view_clip is not None:
            view_indices, abs_indices = view_clip

            view_mask = np.ones(self.view_shape, dtype=np.bool).flatten()
            view_mask[view_indices] = self.mask.flatten()[abs_indices]

            view_map = np.zeros(self.view_shape, dtype=np.int).flatten()
            view_map[view_indices] = self.map.flatten()[abs_indices]
            return self._encoder.encode(view_map, view_mask)
        else:
            return self._encoder.encode(self.map)

    def render_rgb(self, img: np.ndarray):
        img[self.mask] = 8
