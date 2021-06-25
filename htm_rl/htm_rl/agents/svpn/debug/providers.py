from typing import TYPE_CHECKING, Optional, Union

import numpy as np

from htm_rl.agents.svpn.debug.debugger import Debugger
from htm_rl.common.utils import isnone, ensure_list
from htm_rl.experiment import Experiment

if TYPE_CHECKING:
    from htm_rl.agents.svpn.agent import SvpnAgent
    from htm_rl.envs.biogwlab.environment import Environment
    from htm_rl.envs.biogwlab.agent import Agent as EnvAgent


class AgentStateProvider(Debugger):
    env: Environment

    def __init__(self, experiment: Experiment):
        super(AgentStateProvider, self).__init__(experiment)
        self.origin = None

    @property
    def env_agent(self) -> EnvAgent:
        # noinspection PyTypeChecker
        return self.env.entities['agent']

    @property
    def state(self):
        return self.env_agent.position, self.env_agent.view_direction

    def overwrite(self, position=None, view_direction=None):
        if self.origin is None:
            self.origin = self.env_agent.position, self.env_agent.view_direction
        self._set(position, view_direction)

    def restore(self):
        if self.origin is None:
            raise ValueError('Nothing to restore')
        self._set(*self.origin)
        self.origin = None

    def _set(self, position, view_direction):
        self.env_agent.position = isnone(position, self.env_agent.position)
        self.env_agent.view_direction = isnone(view_direction, self.env_agent.view_direction)


class TDErrorProvider(Debugger):
    agent: SvpnAgent

    @property
    def td_error(self):
        return self.agent.sqvn.TD_error


class AnomalyProvider(Debugger):
    agent: SvpnAgent

    @property
    def anomaly(self):
        return self.agent.sa_transition_model.anomaly


class EnvMapProvider(Debugger):
    env: Environment

    _maps: Optional[list[np.ndarray]]
    _titles: list[str]
    name_str: str
    include_observation: bool

    def __init__(self, experiment: Experiment):
        super(EnvMapProvider).__init__(experiment)

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
