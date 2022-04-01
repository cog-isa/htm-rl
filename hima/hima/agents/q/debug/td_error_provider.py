from hima.agents.rnd.debug.debugger import Debugger
from hima.agents.q.agent import QAgent


class TDErrorProvider(Debugger):
    agent: QAgent

    @property
    def td_error(self):
        # FIXME
        return self.agent.TD_error