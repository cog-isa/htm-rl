from typing import TYPE_CHECKING

import numpy as np

from htm_rl.agents.svpn.debug.debugger import Debugger
from htm_rl.agents.svpn.debug.providers import AgentStateProvider
from htm_rl.experiment import Experiment


if TYPE_CHECKING:
    from htm_rl.envs.biogwlab.environment import Environment


class PositionTracker(Debugger):
    fill_value: float = 0.
    name_prefix: str = 'position'

    env: Environment
    agent_state_provider: AgentStateProvider
    heatmap: np.ndarray

    def __init__(self, experiment: Experiment):
        super(PositionTracker, self).__init__(experiment)

        self.agent_state_provider = AgentStateProvider(experiment)
        self.heatmap = np.full(self.env.shape, self.fill_value, dtype=np.float)
        self.agent.set_breakpoint('act', self.on_act)

    def on_act(self, agent, act, *args, **kwargs):
        self.heatmap[self.agent_state_provider.position] += 1
        act(*args, **kwargs)

    def reset(self):
        self.heatmap.fill(self.fill_value)

    @property
    def title(self) -> str:
        return self.filename

    @property
    def filename(self) -> str:
        return f'{self.name_prefix}_{self._default_config_identifier}_{self._default_progress_identifier}'

# class BaseHeatmap:
#     fill_value: float = 0.
#     heatmap: np.ndarray
#
#     def __init__(self, shape: tuple[int, int]):
#         self.heatmap = np.full(shape, self.fill_value, dtype=np.float)
#
#     def update(self, position, value):
#         self.heatmap[position] += value
#
#     def reset(self):
#         self.heatmap.fill(self.fill_value)
