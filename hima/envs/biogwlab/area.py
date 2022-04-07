from typing import Optional

import numpy as np

from hima.envs.biogwlab.environment import Environment
from hima.envs.biogwlab.generation.areas import MultiAreaMapGenerator
from hima.envs.biogwlab.module import Entity, EntityType, Module
from hima.envs.biogwlab.renderer import render_mask
from hima.envs.biogwlab.view_clipper import ViewClip


class Area(Entity):
    family = 'area'
    type = EntityType.Area

    mask: np.ndarray

    def __init__(self, mask: np.ndarray, **entity):
        super(Area, self).__init__(**entity)
        self.mask = mask
        self.initialized = True

    def render(self, view_clip: ViewClip = None):
        return render_mask(self.mask, view_clip)

    def append_mask(self, mask: np.ndarray):
        mask |= self.mask

    def append_position(self, exist: bool, position):
        return exist or self.mask[position]


class MultiAreaGenerator(Module):
    env: Environment
    generator: MultiAreaMapGenerator
    last_seed: Optional[int]

    area_config: dict

    def __init__(self, n_types: int, env: Environment, name: str, **area_entity):
        super(MultiAreaGenerator, self).__init__(name=name)

        self.env = env
        self.generator = MultiAreaMapGenerator(shape=env.shape, n_types=n_types)
        self.last_seed = None
        self.area_config = area_entity

    def generate(self, seeds):
        seed = seeds['map']
        if self.last_seed == seed:
            return

        # Removing previously generated areas is handled by
        # `add_module` - new areas will have the same names
        # and therefore overwrite them.
        self.last_seed = seed
        areas_map = self.generator.generate(seed)
        for i in range(self.generator.n_types):
            mask = areas_map == i
            name = f'{self.name}%{i}'
            area = Area(mask=mask, name=name, **self.area_config)
            self.env.add_module(area)
