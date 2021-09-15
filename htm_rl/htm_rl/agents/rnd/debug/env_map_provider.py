from typing import Optional

import numpy as np

from htm_rl.agents.rnd.debug.debugger import Debugger
from htm_rl.common.utils import ensure_list
from htm_rl.envs.biogwlab.environment import Environment
from htm_rl.scenarios.debug_output import ImageOutput
from htm_rl.scenarios.standard.scenario import Scenario


class EnvMapProvider(Debugger):
    env: Environment

    _maps: Optional[list[np.ndarray]]
    _titles: list[str]
    name_str: str
    include_observation: bool

    def __init__(self, scenario: Scenario):
        super().__init__(scenario)

        self._maps = None
        config = self.scenario.config
        self.name_str = f'map_{config["env"]}_{config["env_seed"]}'
        self._titles = [
            f'{config["env"]}, seed={config["env_seed"]}',
            'agent observation'
        ]

    @property
    def maps(self) -> list[np.ndarray]:
        if self._maps is None:
            self._maps = ensure_list(self.env.render_rgb())
        return self._maps

    @property
    def titles(self):
        titles = self._titles
        if not self.has_observation:
            titles = titles[:1]
        return titles

    @property
    def has_observation(self):
        return len(self.maps) > 1

    def print_map(self, renderer: ImageOutput, n: int = 1):
        if n > 1 and not self.has_observation:
            n = 1
        for i in range(n):
            renderer.handle_img(self.maps[i], self.titles[i])
