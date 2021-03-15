from typing import Dict

import numpy as np

from htm_rl.envs.biogwlab.environment import Environment


class Regenerator:
    base_rates = {
        'map': None, 'food': None, 'agent': None
    }

    seed: int
    seeds: Dict[str, int]
    rates: Dict[str, int]
    episode: int

    def __init__(self, env: Environment, **rates):
        self.seed = env.seed
        self.rates = {**self.base_rates, **rates}
        self.episode = -1
        rng = np.random.default_rng(self.seed)
        self.seeds = {key: rng.integers(1_000_000) for key in self.rates}

    def generate_seeds(self):
        self.episode += 1
        regenerate = False  # marks for regeneration all dependants, note loop order
        for key in ['map', 'food', 'agent']:
            rate = self.rates[key]
            if regenerate or (rate is not None and (self.episode + 1) % rate == 0):
                seed = self.seeds[key]
                rng = np.random.default_rng(seed)
                self.seeds[key] = rng.integers(1_000_000)
                regenerate = True

        return self.seeds
