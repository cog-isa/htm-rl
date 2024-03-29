import numpy as np
from numpy import ma

from htm_rl.agents.rnd.debug.agent_state_provider import AgentStateProvider
from htm_rl.agents.rnd.debug.debugger import Debugger
from htm_rl.common.debug import inject_debug_tools
from htm_rl.envs.biogwlab.environment import Environment
from htm_rl.scenarios.debug_output import ImageOutput
from htm_rl.scenarios.standard.scenario import Scenario


class TrajectoryTracker(Debugger):
    name_prefix: str = 'position'

    env: Environment
    agent_state_provider: AgentStateProvider
    heatmap: ma.MaskedArray

    def __init__(self, scenario: Scenario, act_method_name='act'):
        super(TrajectoryTracker, self).__init__(scenario)

        self.agent_state_provider = AgentStateProvider(scenario)
        self.heatmap = ma.masked_all(self.env.shape, dtype=np.int)

        inject_debug_tools(self.agent)
        # noinspection PyUnresolvedReferences
        self.agent.set_breakpoint(act_method_name, self.on_act)

    def on_act(self, agent, act, *args, **kwargs):
        action = act(*args, **kwargs)
        if action is not None:
            position = self.agent_state_provider.position
            # from htm_rl.envs.biogwlab.move_dynamics import DIRECTIONS_ORDER
            # print(position, DIRECTIONS_ORDER[action])
            if self.heatmap.mask[position]:
                self.heatmap[position] = 0
            self.heatmap[position] += 1
        # else:
        #     print('===================')
        return action

    def reset(self):
        self.heatmap.mask[:] = True

    @property
    def title(self) -> str:
        return self.name_prefix

    @property
    def filename(self) -> str:
        return f'{self.name_prefix}_{self._default_config_identifier}_{self._default_progress_identifier}'

    def print_map(self, renderer: ImageOutput):
        renderer.handle_img(self.heatmap, self.title, with_value_text=True)
        self.reset()
