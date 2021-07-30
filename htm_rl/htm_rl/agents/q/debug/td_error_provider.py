from htm_rl.agents.rnd.debug.debugger import Debugger
from htm_rl.agents.svpn.agent import SvpnAgent


class TDErrorProvider(Debugger):
    agent: SvpnAgent

    @property
    def td_error(self):
        return self.agent.sqvn.TD_error