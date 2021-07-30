from typing import Optional

from htm_rl.agents.rnd.debug.agent_state_provider import AgentStateProvider
from htm_rl.agents.rnd.debug.debugger import Debugger
from htm_rl.agents.q.agent import QAgent
from htm_rl.common.sdr import SparseSdr
from htm_rl.envs.biogwlab.environment import Environment
from htm_rl.envs.biogwlab.module import EntityType
from htm_rl.experiment import Experiment


class StateEncodingProvider(Debugger):
    agent: QAgent
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
                state = self.env.render()
                s = self.agent.sa_encoder.encode_state(state, learn=False)
                encoding_scheme[position] = s

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