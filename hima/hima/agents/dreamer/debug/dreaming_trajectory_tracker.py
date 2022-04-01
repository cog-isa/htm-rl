from typing import Optional

import numpy as np
from numpy import ma

from hima.agents.dreamer.agent import DreamerAgent
from hima.agents.q.debug.state_encoding_provider import StateEncodingProvider
from hima.agents.rnd.debug.debugger import Debugger
from hima.common.debug import inject_debug_tools
from hima.common.sdr import SparseSdr
from hima.envs.biogwlab.environment import Environment
from hima.scenarios.debug_output import ImageOutput
from hima.scenarios.standard.scenario import Scenario


class DreamingTrajectoryTracker(Debugger):
    name_prefix: str = 'dreaming_rollout'

    agent: DreamerAgent
    env: Environment
    state_encoding_provider: StateEncodingProvider
    encoding_scheme: dict[tuple[int, int], SparseSdr]
    heatmap: ma.MaskedArray

    starting_pos: Optional[tuple[int, int]]
    trajectory_len: int

    def __init__(
            self, scenario: Scenario,
            state_encoding_provider: StateEncodingProvider = None
    ):
        super(DreamingTrajectoryTracker, self).__init__(scenario)

        if state_encoding_provider is None:
            state_encoding_provider = StateEncodingProvider(scenario)
        self.state_encoding_provider = state_encoding_provider

        self.heatmap = ma.masked_all(self.env.shape, dtype=np.int)
        self.encoding_scheme = dict()
        self.trajectory_len = 0
        self.starting_pos = None
        self.inject_debug_tools_to_dreamer()

    def inject_debug_tools_to_dreamer(self):
        inject_debug_tools(self.agent.dreamer)
        # noinspection PyUnresolvedReferences
        self.agent.dreamer.set_breakpoint('_put_into_dream', self.on_put_into_dreaming)
        # noinspection PyUnresolvedReferences
        self.agent.dreamer.set_breakpoint('_move_in_dream', self.on_move_in_dream)

    # noinspection PyUnusedLocal
    def on_put_into_dreaming(self, agent, put_into_dream, *args, **kwargs):
        res = put_into_dream(*args, **kwargs)
        self.state_encoding_provider.reset()
        self.encoding_scheme = self.state_encoding_provider.get_encoding_scheme()
        self.starting_pos = self.env.agent.position
        return res

    # noinspection PyUnusedLocal
    def on_move_in_dream(self, agent, move_in_dream, *args, **kwargs):
        state = args[0]
        position, _, _ = self.state_encoding_provider.decode_state(
            state, self.encoding_scheme
        )

        if position is not None:
            if self.heatmap.mask[position]:
                self.heatmap[position] = 0
            self.heatmap[position] += 1

        self.trajectory_len += 1
        return move_in_dream(*args, **kwargs)

    def print_map(self, renderer: ImageOutput, min_traj=1):
        if np.all(self.heatmap.mask):
            return
        if np.sum(self.heatmap) >= min_traj:
            renderer.handle_img(
                self.heatmap, self.title, with_value_text=True
            )
        self.reset()

    def reset(self):
        self.heatmap.mask[:] = True
        self.trajectory_len = 0

    @property
    def title(self) -> str:
        return f'{self.name_prefix}_{self.starting_pos}_{self.trajectory_len}'
