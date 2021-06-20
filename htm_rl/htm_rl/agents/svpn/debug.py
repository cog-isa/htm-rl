from htm_rl.agents.agent import Wrapper
from htm_rl.agents.svpn.agent import SvpnAgent


class ValueProvider(Wrapper):
    root_agent: SvpnAgent

    def get_info(self) -> dict:
        res = self.agent.get_info()

        sa_sdr = self.root_agent._current_sa_sdr
        if sa_sdr is not None:
            res['value'] = self.root_agent.sqvn._value_option(sa_sdr, greedy=True)
            # res['value_exp'] = self.root_agent.sqvn._value_option(sa_sdr, greedy=False)
        return res


class TDErrorProvider(Wrapper):
    root_agent: SvpnAgent

    def get_info(self) -> dict:
        res = self.agent.get_info()
        res['td_error'] = self.root_agent.sqvn.TD_error
        return res


class AnomalyProvider(Wrapper):
    root_agent: SvpnAgent

    def get_info(self) -> dict:
        res = self.agent.get_info()
        res['anomaly'] = self.root_agent.sa_transition_model.anomaly
        return res


class DreamingLengthProvider(Wrapper):
    root_agent: SvpnAgent

    def get_info(self) -> dict:
        res = self.agent.get_info()
        res['dream_length'] = self.root_agent.dream_length
        return res