from typing import Optional

import numpy as np

from htm_rl.agents.svpn.agent import SvpnAgent
from htm_rl.agents.svpn.debug.debugger import Debugger
from htm_rl.common.sdr import SparseSdr
from htm_rl.common.utils import isnone, ensure_list
from htm_rl.envs.biogwlab.agent import Agent as EnvAgent
from htm_rl.envs.biogwlab.environment import Environment
from htm_rl.envs.biogwlab.module import EntityType
from htm_rl.experiment import Experiment


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
        return self.position, self.view_direction

    @property
    def position(self):
        return self.env_agent.position

    @property
    def view_direction(self):
        return self.env_agent.view_direction

    def overwrite(self, position=None, view_direction=None):
        if self.origin is None:
            self.origin = self.state
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


class StateEncodingProvider(Debugger):
    agent: SvpnAgent
    env: Environment

    position_provider: AgentStateProvider

    def __init__(self, experiment: Experiment):
        super().__init__(experiment)
        self.position_provider = AgentStateProvider(experiment)

    def get_encoding_scheme(self) -> dict[tuple[int, int], SparseSdr]:
        height, width = self.env.shape
        obstacle_mask = self.env.aggregated_mask[EntityType.Obstacle]
        encoding_scheme = dict()

        for i in range(height):
            for j in range(width):
                if obstacle_mask[i, j]:
                    continue
                position = i, j
                self.position_provider.overwrite(position)
                observation = self.env.render()
                encoding_scheme[position] = observation

        self.position_provider.restore()
        return encoding_scheme

    @staticmethod
    def decode_state(
            state: SparseSdr,
            encoding_scheme: dict[tuple[int, int], SparseSdr],
            min_overlap_rate: float = .6
    ) -> Optional[tuple[int, int]]:
        best_match = None, 0.
        state = set(state)
        for position, state_ in encoding_scheme.items():
            state_ = set(state_)
            overlap = len(state & state_) / len(state)
            if overlap < min_overlap_rate:
                continue
            if overlap > best_match[1]:
                best_match = position, overlap

        return best_match[0]
