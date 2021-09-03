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

    def __init__(self, scenario: Scenario, act_method_name='act'):
        super(TrajectoryTracker, self).__init__(scenario)

        self.agent_state_provider = AgentStateProvider(scenario)
        self.heatmap = np.full(self.env.shape, self.fill_value, dtype=np.int)
        # noinspection PyUnresolvedReferences
        self.agent.set_breakpoint(act_method_name, self.on_act)

    def on_act(self, agent, act, *args, **kwargs):
        action = act(*args, **kwargs)
        if action is not None:
            # if not agent.train:
            #     from htm_rl.envs.biogwlab.move_dynamics import DIRECTIONS_ORDER
            #     print(self.agent_state_provider.position, DIRECTIONS_ORDER[action])
            self.heatmap[self.agent_state_provider.position] += 1
        # else:
        #     print('===================')
        return action

    def reset(self):
        self.heatmap.fill(self.fill_value)

    @property
    def title(self) -> str:
        return self.filename

    @property
    def filename(self) -> str:
        return f'{self.name_prefix}_{self._default_config_identifier}_{self._default_progress_identifier}'