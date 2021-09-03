import numpy as np

from htm_rl.agents.rnd.debug.agent_state_provider import AgentStateProvider
from htm_rl.agents.rnd.debug.debugger import Debugger
from htm_rl.envs.biogwlab.environment import Environment
from htm_rl.scenarios.standard.scenario import Scenario


class TrajectoryTracker(Debugger):
    fill_value: float = 0.
    name_prefix: str = 'position'

    env: Environment
    agent_state_provider: AgentStateProvider
    heatmap: np.ndarray

    # noinspection PyUnresolvedReferences
    def __init__(self, experiment: Scenario, act_method_name='act'):
        super(TrajectoryTracker, self).__init__(experiment)

        self.agent_state_provider = AgentStateProvider(experiment)
        self.heatmap = np.full(self.env.shape, self.fill_value, dtype=np.int)
        self.agent.set_breakpoint(act_method_name, self.on_act)

    def on_act(self, agent, act, *args, **kwargs):
        action = act(*args, **kwargs)
        if action is not None:
            self.heatmap[self.agent_state_provider.position] += 1
        return action

    def reset(self):
        self.heatmap.fill(self.fill_value)

    @property
    def title(self) -> str:
        return self.filename

    @property
    def filename(self) -> str:
        return f'{self.name_prefix}_{self._default_config_identifier}_{self._default_progress_identifier}'