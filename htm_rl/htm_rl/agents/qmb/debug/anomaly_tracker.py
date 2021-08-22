import numpy as np

from htm_rl.agents.qmb.debug.anomaly_provider import AnomalyProvider
from htm_rl.agents.rnd.debug.agent_state_provider import AgentStateProvider
from htm_rl.agents.rnd.debug.debugger import Debugger
from htm_rl.envs.biogwlab.environment import Environment
from htm_rl.scenarios.standard.experiment import Experiment


class AnomalyTracker(Debugger):
    fill_value: float = 1.
    name_prefix: str = 'anomaly'

    env: Environment
    agent_state_provider: AgentStateProvider
    anomaly_provider: AnomalyProvider
    heatmap: np.ndarray
    anomalies: list[float]
    reward_anomalies: list[float]

    def __init__(self, experiment: Experiment):
        super(AnomalyTracker, self).__init__(experiment)

        self.agent_state_provider = AgentStateProvider(experiment)
        self.anomaly_provider = AnomalyProvider(experiment)
        self.heatmap = np.full(self.env.shape, self.fill_value, dtype=np.float)
        self.anomalies = []
        self.reward_anomalies = []
        self.agent.set_breakpoint('act', self.on_act)

    def on_act(self, agent, act, *args, **kwargs):
        action = act(*args, **kwargs)
        anomaly = 1. - self.anomaly_provider.recall
        self.heatmap[self.agent_state_provider.position] = anomaly
        self.anomalies.append(anomaly)

        self.reward_anomalies.append(self.anomaly_provider.reward_anomaly)
        return action

    def reset(self):
        self.heatmap.fill(self.fill_value)

    @property
    def title(self) -> str:
        return self.filename

    @property
    def filename(self) -> str:
        return f'{self.name_prefix}_{self._default_config_identifier}_{self._default_progress_identifier}'