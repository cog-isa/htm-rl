from typing import Optional

from htm_rl.envs.biogwlab.entity import Entity
from htm_rl.envs.biogwlab.generation.obstacles import ObstaclesGenerator


class Obstacles(Entity):
    entity = 'obstacles'

    _generator: ObstaclesGenerator
    _last_seed: Optional[int]

    def __init__(self, density, **obstacles):
        super(Obstacles, self).__init__(**obstacles)
        self._generator = ObstaclesGenerator(shape=self.shape, density=density)
        self._last_seed = None

    def generate(self, seed):
        if self._last_seed is not None and self._last_seed == seed:
            return

        mask = self._generator.generate(seed)
        self.set(
            mask=mask,
            map=None
        )