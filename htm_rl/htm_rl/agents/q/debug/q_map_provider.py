from typing import Optional

import numpy as np

from htm_rl.agents.q.debug.state_encoding_provider import StateEncodingProvider
from htm_rl.agents.rnd.debug.debugger import Debugger
from htm_rl.agents.q.agent import QAgent
from htm_rl.envs.biogwlab.environment import Environment
from htm_rl.scenarios.standard.scenario import Scenario


# noinspection PyPep8Naming
class QMapProvider(Debugger):
    fill_value: float = 0.
    name_prefix: str = 'Q'

    agent: QAgent
    env: Environment

    state_encoding_provider: StateEncodingProvider

    Q: Optional[np.ndarray]

    def __init__(self, scenario: Scenario):
        super().__init__(scenario)
        self.state_encoding_provider = StateEncodingProvider(scenario)
        self.Q = None

    def precompute(self):
        encoding_scheme = self.state_encoding_provider.get_encoding_scheme()

        shape = self.env.shape + (self.env.n_actions,)
        self.Q = np.full(shape, self.fill_value, dtype=np.float)

        for position, s in encoding_scheme.items():
            # noinspection PyProtectedMember
            actions_sa_sdr = self.agent._encode_state_actions(s, learn=False)
            self.Q[position] = self.agent.Q.values(actions_sa_sdr)

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
