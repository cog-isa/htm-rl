from htm_rl.agents.rnd.debug.debugger import Debugger
from htm_rl.agents.qmb.agent import QModelBasedAgent


class AnomalyProvider(Debugger):
    agent: QModelBasedAgent

    @property
    def anomaly(self):
        return self.agent.sa_transition_model.anomaly

    @property
    def recall(self):
        return self.agent.sa_transition_model.recall