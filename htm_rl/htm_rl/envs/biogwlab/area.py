from typing import Tuple, Optional

import numpy as np

from htm_rl.common.sdr_encoders import IntArrayEncoder
from htm_rl.envs.biogwlab.environment import Environment
from htm_rl.envs.biogwlab.generation.areas import MultiAreaMapGenerator
from htm_rl.envs.biogwlab.module import Entity, EntityType, Module


class Area(Entity):
    family = 'area'
    type = EntityType.Area

    mask: np.ndarray

    def __init__(self, mask: np.ndarray,  **entity):
        super(Area, self).__init__(**entity)
        self.mask = mask

    def append_mask(self, mask: np.ndarray):
        mask |= self.mask

    def append_position(self, exist: bool, position):
        return exist or self.mask[position]


class MultiAreaGenerator(Module):
    env: Environment
    generator: MultiAreaMapGenerator
    last_seed: Optional[int]

    def __init__(self, n_types: int, env: Environment, **entity):
        super(MultiAreaGenerator, self).__init__(**entity)

        self.env = env
        self.generator = MultiAreaMapGenerator(shape=env.shape, n_types=n_types)
        self.last_seed = None

    def generate(self, seeds):
        # TODO fix seeding
        seed = seeds['??']
        if self.last_seed == seed:
            return

        areas_map = self.generator.generate(seed)
        for i in range(self.generator.n_types):
            mask = areas_map == i
            name = f'{self.name}%{i}'
            area = Area(mask=mask, name=name)
            self.env.add_module(area)


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
