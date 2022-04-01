from typing import Optional

import numpy as np
from numpy import ma

from hima.agents.q.debug.state_encoding_provider import StateEncodingProvider
from hima.agents.rnd.debug.debugger import Debugger
from hima.agents.q.agent import QAgent
from hima.envs.biogwlab.environment import Environment
from hima.scenarios.debug_output import ImageOutput
from hima.scenarios.standard.scenario import Scenario


# noinspection PyPep8Naming
class QMapProvider(Debugger):
    name_prefix: str = 'Q'

    agent: QAgent
    env: Environment

    state_encoding_provider: StateEncodingProvider

    def __init__(
            self, scenario: Scenario,
            state_encoding_provider: StateEncodingProvider = None
    ):
        super(QMapProvider, self).__init__(scenario)
        if state_encoding_provider is None:
            state_encoding_provider = StateEncodingProvider(scenario)
        self.state_encoding_provider = state_encoding_provider

    def Q(self) -> ma.MaskedArray:
        shape = self.env.shape + (self.env.n_actions,)
        Q = ma.masked_all(shape, dtype=np.float)

        encoding_scheme = self.state_encoding_provider.get_encoding_scheme()
        for position, s in encoding_scheme.items():
            # noinspection PyProtectedMember
            actions_sa_sdr = self.agent._encode_s_actions(s, learn=False)
            Q[position] = self.agent.Q.values(actions_sa_sdr)

        return Q

    @staticmethod
    def V(Q: ma.MaskedArray):
        return Q.max(axis=-1)

    def reshape_q_for_rendering(self, Q):
        assert self.env.n_actions == 4

        shape = Q.shape[0] * 2, Q.shape[1] * 2
        Qr = np.empty(shape, dtype=np.float)
        Qr[0::2, 0::2] = Q[..., 3]
        Qr[0::2, 1::2] = Q[..., 0]
        Qr[1::2, 1::2] = Q[..., 1]
        Qr[1::2, 0::2] = Q[..., 2]
        return Qr

    # noinspection PyPep8Naming
    def print_maps(self, renderer: ImageOutput, q: bool, v: bool):
        Q = self.Q()
        V = self.V(Q)
        if v:
            renderer.handle_img(V, 'V', with_value_text=True)
        if q:
            Q = Q.filled(.0)
            renderer.handle_img(Q, 'Q', with_value_text=False)
