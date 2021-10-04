import numpy as np
from numpy import ma

from htm_rl.agents.q.debug.state_encoding_provider import StateEncodingProvider
from htm_rl.agents.qmb.agent import QModelBasedAgent
from htm_rl.agents.rnd.debug.agent_state_provider import AgentStateProvider
from htm_rl.agents.rnd.debug.debugger import Debugger
from htm_rl.envs.biogwlab.environment import Environment
from htm_rl.scenarios.debug_output import ImageOutput
from htm_rl.scenarios.standard.scenario import Scenario


class AnomalyMapProvider(Debugger):
    fill_value: float = 1.
    name_prefix: str = 'anomaly'

    env: Environment
    agent: QModelBasedAgent
    agent_state_provider: AgentStateProvider
    state_encoding_provider: StateEncodingProvider

    def __init__(
            self, scenario: Scenario,
            state_encoding_provider: StateEncodingProvider = None
    ):
        super(AnomalyMapProvider, self).__init__(scenario)

        self.agent_state_provider = AgentStateProvider(scenario)
        if state_encoding_provider is None:
            state_encoding_provider = StateEncodingProvider(scenario)
        self.state_encoding_provider = state_encoding_provider

    def anomaly(self):
        anomaly_model = self.agent.anomaly_model
        anomaly_pos = ma.masked_all(self.env.shape, dtype=np.float)

        encoding_scheme = self.state_encoding_provider.get_encoding_scheme()
        for position, s in encoding_scheme.items():
            anomaly_pos[position] = anomaly_model.state_anomaly(s)
        return anomaly_pos

    @property
    def title(self) -> str:
        return self.name_prefix

    def print_map(self, renderer: ImageOutput):
        renderer.handle_img(self.anomaly(), self.title, with_value_text=True)
