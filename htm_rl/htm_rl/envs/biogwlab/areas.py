from typing import Tuple, Optional

import numpy as np

from htm_rl.common.sdr_encoders import IntArrayEncoder
from htm_rl.envs.biogwlab.module import Entity
from htm_rl.envs.biogwlab.generation.areas import AreasGenerator


class Areas(Entity):
    entity = 'areas'

    _generator: AreasGenerator
    _last_seed: Optional[int]

    def __init__(self, **areas):
        super(Areas, self).__init__(**areas)
        self._generator = AreasGenerator(shape=self.shape, n_types=self.n_types)
        self._last_seed = None

    def generate(self, seeds):
        seed = seeds['map']
        if self._last_seed is not None and self._last_seed == seed:
            return

        self.set(
            mask=None,
            map_=self._generator.generate(seed)
        )


class AreasRenderer:
    """TODO use or remove it"""
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
