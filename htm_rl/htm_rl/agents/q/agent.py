from typing import Optional, List

import numpy as np
from numpy.random import Generator

from htm_rl.agents.agent import Agent
from htm_rl.agents.q.eligibility_traces import EligibilityTraces
from htm_rl.agents.q.qvn import QValueNetwork
from htm_rl.agents.q.sa_encoder import SaEncoder
from htm_rl.common.sdr import SparseSdr
from htm_rl.common.utils import exp_decay
from htm_rl.envs.env import Env


class QAgent(Agent):
    n_actions: int
    sa_encoder: SaEncoder
    Q: QValueNetwork
    E_traces: Optional[EligibilityTraces]

    exploration_eps: Optional[tuple[float, float]]
    softmax_enabled: bool

    _current_sa_sdr: Optional[SparseSdr]
    _rng: Generator

    def __init__(
            self,
            env: Env,
            seed: int,
            sa_encoder: dict,
            qvn: dict,
            eligibility_traces: dict = None,
            exploration_eps: tuple[float, float] = None,
            softmax_enabled: bool = False
    ):
        self.n_actions = env.n_actions
        self.sa_encoder = SaEncoder(env, seed, **sa_encoder)
        self.Q = QValueNetwork(self.sa_encoder.output_sdr_size, seed, **qvn)
        self.E_traces = EligibilityTraces(
            self.sa_encoder.output_sdr_size,
            **eligibility_traces
        )
        self.exploration_eps = exploration_eps
        self.softmax_enabled = softmax_enabled

        self._rng = np.random.default_rng(seed)
        self._current_sa_sdr = None

    @property
    def name(self):
        return 'q'

    @property
    def td_lambda(self):
        return self.E_traces is not None

    def act(self, reward: float, state: SparseSdr, first: bool):
        if first:
            self.on_new_episode()

        s = self.sa_encoder.encode_state(state, learn=True)
        actions_sa_sdr = self.sa_encoder.encode_actions(s, learn=True)

        greedy_action = self.choose(actions_sa_sdr)
        greedy_sa_sdr = actions_sa_sdr[greedy_action]

        if not first:
            # process feedback
            self.Q.update(
                sa=self._current_sa_sdr, reward=reward,
                sa_next=greedy_sa_sdr,
                E_traces=self.E_traces.E
            )

        self._current_sa_sdr = greedy_sa_sdr
        return greedy_action

    # noinspection PyTypeChecker
    def choose(self, actions: List[SparseSdr]) -> int:
        if self.exploration_eps is not None:
            if self._rng.random() < self.exploration_eps[0]:
                return self._rng.integers(self.n_actions)

        action_values = self.Q.values(actions)
        if self.softmax_enabled:
            return self._rng.choice(self.n_actions, p=softmax(action_values))
        return np.argmax(action_values)

    def on_new_episode(self):
        self.Q.decay_learning_factors()
        self.E_traces.reset()
        if self.exploration_eps is not None:
            exp_decay(self.exploration_eps)


def softmax(x):
    """Computes softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)
