import numpy as np

from htm_rl.agents.svpn.debug.debugger import Debugger
from htm_rl.agents.svpn.debug.providers import AgentStateProvider, AnomalyProvider
from htm_rl.envs.biogwlab.environment import Environment
from htm_rl.experiment import Experiment


class AnomalyTracker(Debugger):
    fill_value: float = 1.
    name_prefix: str = 'anomaly'

    env: Environment
    agent_state_provider: AgentStateProvider
    anomaly_provider: AnomalyProvider
    heatmap: np.ndarray

    def __init__(self, experiment: Experiment):
        super(AnomalyTracker, self).__init__(experiment)

        self.agent_state_provider = AgentStateProvider(experiment)
        self.anomaly_provider = AnomalyProvider(experiment)
        self.heatmap = np.full(self.env.shape, self.fill_value, dtype=np.float)
        self.agent.set_breakpoint('act', self.on_act)

    def on_act(self, agent, act, *args, **kwargs):
        self.heatmap[self.agent_state_provider.position] += self.anomaly_provider.anomaly
        act(*args, **kwargs)

    def reset(self):
        self.heatmap.fill(self.fill_value)

    @property
    def title(self) -> str:
        return self.filename

    @property
    def filename(self) -> str:
        return f'{self.name_prefix}_{self._default_config_identifier}_{self._default_progress_identifier}'
