from typing import Optional, Tuple

import numpy as np

from htm_rl.envs.biogwlab.environment import Environment
from htm_rl.envs.biogwlab.module import Entity, EntityType
from htm_rl.envs.biogwlab.generation.obstacles import ObstacleMaskGenerator
from htm_rl.envs.biogwlab.renderer import render_mask
from htm_rl.envs.biogwlab.view_clipper import ViewClip


class Obstacle(Entity):
    family = 'obstacle'
    type = EntityType.Obstacle

    generator: ObstacleMaskGenerator
    mask: np.ndarray
    last_seed: Optional[int]

    def __init__(self, env: Environment, density, **entity):
        super(Obstacle, self).__init__(**entity)

        self.generator = ObstacleMaskGenerator(shape=env.shape, density=density)
        self.mask = np.zeros(env.shape, dtype=np.bool)
        self.last_seed = None

    def generate(self, seeds):
        seed = seeds['map']
        if self.last_seed == seed:
            return

        self.mask = self.generator.generate(seed)
        self.initialized = True

    def render(self, view_clip: ViewClip = None):
        return render_mask(self.mask, view_clip)

    def append_mask(self, mask: np.ndarray):
        mask |= self.mask

    def append_position(self, exist: bool, position):
        return exist or self.mask[position]


class BorderObstacle(Entity):
    family = 'obstacle.border'
    type = EntityType.Obstacle

    shape: Tuple[int, int]

    def __init__(self, env: Environment, rendering=True, **entity):
        super(BorderObstacle, self).__init__(rendering=rendering, **entity)

        self.shape = env.shape
        self.initialized = True

    def render(self, view_clip: ViewClip = None):
        if view_clip is None:
            # it's not needed without observation window
            # won't be included because of zero size
            return None, 0

        view_shape, abs_indices, view_indices = view_clip

        clipped_mask = np.ones(view_shape, dtype=np.bool).flatten()
        clipped_mask[view_indices] = 0
        return np.flatnonzero(clipped_mask), clipped_mask.size

    def append_mask(self, mask: np.ndarray):
        ...

    def append_position(self, exist: bool, position):
        return exist or not (
            0 <= position[0] < self.shape[0]
            and 0 <= position[1] < self.shape[1]
        )

