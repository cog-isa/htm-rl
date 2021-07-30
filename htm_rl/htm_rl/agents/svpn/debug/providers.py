from typing import Optional

import numpy as np

from htm_rl.agents.q.debug.state_encoding_provider import StateEncodingProvider
from htm_rl.agents.svpn.agent import SvpnAgent
from htm_rl.agents.rnd.debug.debugger import Debugger
from htm_rl.envs.biogwlab.environment import Environment
from htm_rl.experiment import Experiment


# noinspection PyPep8Naming
class ValueMapProvider(Debugger):
    fill_value: float = 0.
    name_prefix: str = 'position'

    agent: SvpnAgent
    env: Environment

    state_encoding_provider: StateEncodingProvider

    Q: Optional[np.ndarray]
    UCB: Optional[np.ndarray]

    def __init__(self, experiment: Experiment):
        super().__init__(experiment)
        self.state_encoding_provider = StateEncodingProvider(experiment)
        self.Q = None
        self.UCB = None

    # noinspection PyProtectedMember
    def precompute(self, greedy_map: bool = False, ucb_map: bool = False):
        encoding_scheme = self.state_encoding_provider.get_encoding_scheme()

        self.Q = None
        self.UCB = None
        if not greedy_map and not ucb_map:
            return

        shape = self.env.shape + (self.env.n_actions,)
        if greedy_map:
            self.Q = np.full(shape, self.fill_value, dtype=np.float)
        if ucb_map:
            self.UCB = np.full(shape, self.fill_value, dtype=np.float)

        for position, state in encoding_scheme.items():
            actions_sa_sdr = self.agent._encode_actions(state, learn=False)
            if greedy_map:
                self.Q[position] = self.agent.sqvn.evaluate_options(actions_sa_sdr)
            if ucb_map:
                self.UCB[position] = self.agent.sqvn.evaluate_options_ucb_term(actions_sa_sdr)

    @staticmethod
    def V(Q) -> np.ndarray:
        return np.max(Q, axis=-1)

    def reshape_q_for_rendering(self, Q):
        assert self.env.n_actions == 4

        shape = Q.shape[0] * 2, Q.shape[1] * 2
        Qr = np.empty(shape, dtype=np.float)
        Qr[0::2, 0::2] = Q[..., 3]
        Qr[0::2, 1::2] = Q[..., 0]
        Qr[1::2, 1::2] = Q[..., 1]
        Qr[1::2, 0::2] = Q[..., 2]
        return Qr
