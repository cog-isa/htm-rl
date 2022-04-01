from typing import Optional

import numpy as np

from hima.agents.rnd.debug.debugger import Debugger
from hima.common.debug import inject_debug_tools
from hima.common.utils import ensure_list, isnone
from hima.envs.biogwlab.environment import Environment
from hima.envs.biogwlab.modules.regenerator import Regenerator
from hima.scenarios.debug_output import ImageOutput
from hima.scenarios.standard.scenario import Scenario


class EnvMapProvider(Debugger):
    env: Environment

    _regenerator: Optional[Regenerator]
    _maps: Optional[list[np.ndarray]]
    _titles: list[str]
    name_str: str
    with_outer_walls: bool
    _last_seeds_tuple: tuple[int]

    def __init__(self, scenario: Scenario, with_outer_walls=True):
        super().__init__(scenario)

        self._maps = None
        self._regenerator = None
        config = self.scenario.config
        self.name_str = f'map_{config["env"]}_{config["env_seed"]}'
        self._titles = [
            f'{config["env"]}, seed={config["env_seed"]}',
            'agent observation'
        ]
        self.with_outer_walls = with_outer_walls

    @property
    def maps(self) -> list[np.ndarray]:
        if self._regenerator is None:
            self._regenerator = self.env.get_module('regenerate')

        current_seeds_tuple = tuple(self._regenerator.seeds.values())
        maps = self._maps

        if self._maps is None or self._last_seeds_tuple != current_seeds_tuple:
            self._maps = ensure_list(self.env.render_rgb(
                show_outer_walls=self.with_outer_walls
            ))
            self._last_seeds_tuple = current_seeds_tuple

        maps = isnone(maps, self._maps)
        return maps

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
