from typing import Optional

from hima.agents.rnd.debug.agent_state_provider import AgentStateProvider
from hima.agents.rnd.debug.debugger import Debugger
from hima.agents.q.agent import QAgent
from hima.common.sdr import SparseSdr
from hima.common.utils import Coord2d
from hima.envs.biogwlab.environment import Environment
from hima.envs.biogwlab.module import EntityType
from hima.scenarios.standard.scenario import Scenario


class StateEncodingProvider(Debugger):
    agent: QAgent
    env: Environment

    position_provider: AgentStateProvider
    encoding_scheme: Optional[dict]

    def __init__(self, scenario: Scenario):
        super().__init__(scenario)
        self.position_provider = AgentStateProvider(scenario)
        self.encoding_scheme = None

    def reset(self):
        self.encoding_scheme = None

    def get_encoding_scheme(self) -> dict[tuple[int, int], SparseSdr]:
        if self.encoding_scheme is not None:
            return self.encoding_scheme

        height, width = self.env.shape
        obstacle_mask = self.env.aggregated_mask[EntityType.Obstacle]
        encoding_scheme = {}

        for i in range(height):
            for j in range(width):
                if obstacle_mask[i, j]:
                    continue
                position = i, j
                self.position_provider.overwrite(position)
                state = self.env.render()
                s = self.agent.sa_encoder.encode_state(state, learn=False)
                encoding_scheme[position] = s

        self.position_provider.restore()
        self.encoding_scheme = encoding_scheme
        return encoding_scheme

    @staticmethod
    def decode_state(
            state: SparseSdr,
            encoding_scheme: dict[Coord2d, SparseSdr],
            min_overlap_rate: float = .5
    ) -> tuple[Optional[Coord2d], float, Optional[SparseSdr]]:
        best_match = None, 0., None
        state = set(state)
        for position, state_ in encoding_scheme.items():
            state_ = set(state_)
            overlap = len(state & state_) / len(state_)
            if overlap < min_overlap_rate:
                continue
            if overlap > best_match[1]:
                best_match = position, overlap, state_

        return best_match
