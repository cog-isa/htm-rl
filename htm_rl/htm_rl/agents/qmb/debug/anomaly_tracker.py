import numpy as np
from numpy import ma

from htm_rl.agents.qmb.debug.anomaly_provider import AnomalyProvider
from htm_rl.agents.rnd.debug.agent_state_provider import AgentStateProvider
from htm_rl.agents.rnd.debug.debugger import Debugger
from htm_rl.envs.biogwlab.environment import Environment
from htm_rl.scenarios.standard.scenario import Scenario


class AnomalyTracker(Debugger):
    fill_value: float = 1.
    name_prefix: str = 'anomaly'

    env: Environment
    agent_state_provider: AgentStateProvider
    anomaly_provider: AnomalyProvider
    heatmap: ma.MaskedArray

    keep_anomalies: bool

    def __init__(self, scenario: Scenario, keep_anomalies=True):
        super(AnomalyTracker, self).__init__(scenario)

        self.agent_state_provider = AgentStateProvider(scenario)
        self.anomaly_provider = AnomalyProvider(scenario)
        # self.heatmap = np.full(self.env.shape, self.fill_value, dtype=np.float)
        self.heatmap = ma.masked_all(self.env.shape, dtype=np.float)
        self.keep_anomalies = keep_anomalies
        if self.keep_anomalies:
            self.anomalies = []
            self.reward_anomalies = []
        # noinspection PyUnresolvedReferences
        self.agent.set_breakpoint('act', self.on_act)

    def on_act(self, agent, act, *args, **kwargs):
        action = act(*args, **kwargs)
        if action is None:
            return action

        anomaly = 1. - self.anomaly_provider.recall
        self.heatmap[self.agent_state_provider.position] = anomaly

        if self.keep_anomalies:
            self.anomalies.append(anomaly)
            self.reward_anomalies.append(self.anomaly_provider.reward_anomaly)
        return action

    def reset(self):
        self.heatmap.mask[:] = True

    @property
    def title(self) -> str:
        return self.name_prefix

    @property
    def filename(self) -> str:
        return f'{self.name_prefix}_{self._default_config_identifier}_{self._default_progress_identifier}'