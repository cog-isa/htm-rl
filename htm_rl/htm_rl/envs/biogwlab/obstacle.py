from typing import Optional, Tuple

import numpy as np

from htm_rl.envs.biogwlab.environment import Environment
from htm_rl.envs.biogwlab.module import Entity, EntityType
from htm_rl.envs.biogwlab.generation.obstacles import ObstacleMaskGenerator


class Obstacle(Entity):
    family = 'obstacle'
    type = EntityType.Obstacle

    generator: ObstacleMaskGenerator
    mask: np.ndarray
    last_seed: Optional[int]

    def __init__(self, env: Environment, density, **entity):
        super(Obstacle, self).__init__(**entity)

        self.generator = ObstacleMaskGenerator(shape=env.shape, density=density)
        self.mask = np.zeros(env.shape, dtype=np.int)
        self.last_seed = None

    def generate(self, seeds):
        seed = seeds['map']
        if self.last_seed == seed:
            return

        self.mask = self.generator.generate(seed)

    def append_mask(self, mask: np.ndarray):
        mask |= self.mask

    def append_position(self, exist: bool, position):
        return exist or self.mask[position]


class BorderObstacle(Entity):
    family = 'obstacle.border'
    type = EntityType.Obstacle

    shape: Tuple[int, int]

    def __init__(self, env: Environment, **entity):
        super(BorderObstacle, self).__init__(**entity)

        self.shape = env.shape

    def append_mask(self, mask: np.ndarray):
        ...

    def append_position(self, exist: bool, position):
        return exist or not (
            0 <= position[0] < self.shape[0]
            and 0 <= position[1] < self.shape[1]
        )

