import numpy as np
from numpy import ma

from htm_rl.agents.dreamer.agent import DreamerAgent
from htm_rl.agents.dreamer.dreaming_double import DreamingDouble
from htm_rl.agents.rnd.debug.agent_state_provider import AgentStateProvider
from htm_rl.agents.rnd.debug.debugger import Debugger
from htm_rl.common.debug import inject_debug_tools
from htm_rl.envs.biogwlab.environment import Environment
from htm_rl.scenarios.debug_output import ImageOutput
from htm_rl.scenarios.standard.scenario import Scenario


class DreamingHeatmapTracker(Debugger):
    fill_value: float = 0.
    name_prefix = 'dreaming_heatmap'
    dreaming_heatmap: ma.MaskedArray

    env: Environment
    agent: DreamerAgent
    agent_state_provider: AgentStateProvider

    def __init__(self, scenario: Scenario):
        super(DreamingHeatmapTracker, self).__init__(scenario)

        self.agent_state_provider = AgentStateProvider(scenario)
        self.dreaming_heatmap = ma.masked_all(self.env.shape, dtype=np.int)

        dreamer = self.agent.dreamer
        inject_debug_tools(dreamer)
        # noinspection PyUnresolvedReferences
        dreamer.set_breakpoint('dream', self.on_dream)

    def on_dream(self, dreaming_double: DreamingDouble, dream, *args, **kwargs):
        position = self.agent_state_provider.position

        if self.dreaming_heatmap.mask[position]:
            self.dreaming_heatmap[position] = 0
        self.dreaming_heatmap[position] += 1

        dream(*args, **kwargs)

    def reset(self):
        self.dreaming_heatmap.mask[:] = True

    @property
    def title(self) -> str:
        return self.name_prefix

    def print_map(self, renderer: ImageOutput):
        if np.all(self.dreaming_heatmap.mask):
            return

        renderer.handle_img(
            self.dreaming_heatmap, self.title, with_value_text=True
        )
        self.reset()
