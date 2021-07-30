from typing import Optional

import numpy as np

from htm_rl.agents.rnd.debug.debugger import Debugger
from htm_rl.common.utils import ensure_list
from htm_rl.envs.biogwlab.environment import Environment
from htm_rl.experiment import Experiment


class EnvMapProvider(Debugger):
    env: Environment

    _maps: Optional[list[np.ndarray]]
    _titles: list[str]
    name_str: str
    include_observation: bool

    def __init__(self, experiment: Experiment):
        super().__init__(experiment)

        self._maps = None
        config = self.experiment.config
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