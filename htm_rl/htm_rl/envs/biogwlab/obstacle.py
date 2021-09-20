from typing import Optional

import numpy as np

from htm_rl.envs.biogwlab.env_shape_params import EnvShapeParams
from htm_rl.envs.biogwlab.environment import Environment
from htm_rl.envs.biogwlab.generation.obstacles import ObstacleMaskGenerator, ObstacleMask, ObstacleMaskManual
from htm_rl.envs.biogwlab.module import Entity, EntityType
from htm_rl.envs.biogwlab.renderer import render_mask
from htm_rl.envs.biogwlab.view_clipper import ViewClip


class Obstacle(Entity):
    family = 'obstacle'
    type = EntityType.Obstacle

    generator: ObstacleMask
    shape: EnvShapeParams
    mask: np.ndarray
    last_seed: Optional[int]

    def __init__(self, env: Environment, density=None, map_name=None, **entity):
        super(Obstacle, self).__init__(**entity)

        self.shape = env.renderer.shape
        env_shape = self.shape.env_shape
        if map_name is not None:
            self.generator = ObstacleMaskManual(shape=env_shape, map_name=map_name)
        else:
            self.generator = ObstacleMaskGenerator(shape=env_shape, density=density)

        # init with ones to make outer walls
        self.mask = np.ones(self.shape.full_shape, dtype=np.bool)
        self.last_seed = None

    def generate(self, seeds):
        seed = seeds['map']
        if self.last_seed == seed:
            return

        self.initialized = False
        # generates obstacles only for the effective env shape
        # outer walls are kept unchanged
        inner_area_mask = self.generator.generate(seed)
        self.shape.set_inner_area(self.mask, inner_area_mask)
        self.initialized = True

    def render(self, view_clip: ViewClip = None):
        return render_mask(self.mask, view_clip)

    def append_mask(self, mask: np.ndarray):
        if self.initialized:
            mask |= self.mask

    def append_position(self, exist: bool, position):
        return exist or (self.initialized and self.mask[position])


class BorderObstacle(Entity):
    family = 'obstacle.border'
    type = EntityType.Obstacle

    shape: tuple[int, int]

    def __init__(self, env: Environment, rendering=True, **entity):
        super(BorderObstacle, self).__init__(rendering=rendering, **entity)

        self.shape = env.shape
        self.initialized = True

    def render(self, view_clip: ViewClip = None):
        if view_clip is None:
            # it's not needed without observation window
            # won't be included because of zero size
            return None, 0

        clipped_mask = np.ones(view_clip.shape, dtype=np.bool).flatten()
        clipped_mask[view_clip.view_indices] = 0
        return np.flatnonzero(clipped_mask), clipped_mask.size

    def append_mask(self, mask: np.ndarray):
        ...

    def append_position(self, exist: bool, position):
        return exist or not (
            0 <= position[0] < self.shape[0]
            and 0 <= position[1] < self.shape[1]
        )

