from typing import Optional

import numpy as np

from hima.envs.biogwlab.environment import Environment
from hima.envs.biogwlab.module import Module


class Regenerator(Module):
    base_rates = {
        'map': None, 'food': None, 'agent': None
    }
    base_modulo: int = 1_000_000

    seed: int
    seeds: dict[str, int]
    rates: dict[str, int]
    episode: int

    def __init__(self, env: Environment, name: str, **rates):
        super().__init__(name=name)

        self.seed = env.seed
        self.rates = {**self.base_rates, **rates}
        self.episode = -1

        rng = np.random.default_rng(self.seed)
        self.seeds = {key: rng.integers(self.base_modulo) for key in self.rates}

    def generate_seeds(self):
        def scheduled(rate: Optional[int]):
            if rate is None:
                return False
            return (self.episode + 1) % rate == 0

        self.episode += 1
        regenerate = False  # marks for regeneration all dependants, note loop order
        for key in ['map', 'food', 'agent']:
            if regenerate or scheduled(self.rates[key]):
                seed = self.seeds[key]
                # use the current seed to generate the next seed
                rng = np.random.default_rng(seed)
                self.seeds[key] = rng.integers(self.base_modulo)

                # every dependant modules should be regenerated too
                regenerate = True

        return self.seeds
