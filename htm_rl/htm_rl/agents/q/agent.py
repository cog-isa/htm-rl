from typing import Optional, List

import numpy as np
from numpy.random import Generator

from htm_rl.agents.agent import Agent
from htm_rl.agents.q.eligibility_traces import EligibilityTraces
from htm_rl.agents.q.qvn import QValueNetwork
from htm_rl.agents.q.sa_encoders import SaEncoder
from htm_rl.common.sdr import SparseSdr
from htm_rl.common.utils import exp_decay
from htm_rl.envs.env import Env


class QAgent(Agent):
    n_actions: int
    sa_encoder: SaEncoder
    Q: QValueNetwork
    E_traces: EligibilityTraces

    exploration_eps: Optional[tuple[float, float]]
    softmax_enabled: bool

    _current_sa_sdr: Optional[SparseSdr]
    _rng: Generator

    def __init__(
            self,
            env: Env,
            seed: int,
            qvn: dict,
            sa_encoder: dict = None,
            eligibility_traces: dict = None,
            exploration_eps: tuple[float, float] = None,
            softmax_enabled: bool = False
    ):
        self.n_actions = env.n_actions
        self.sa_encoder = make_sa_encoder(env, seed, sa_encoder)
        sa_sdr_size = self.sa_encoder.output_sdr_size

        self.Q = QValueNetwork(sa_sdr_size, seed, **qvn)
        self.E_traces = EligibilityTraces(
            sa_sdr_size,
            **eligibility_traces
        )
        self.exploration_eps = exploration_eps
        self.softmax_enabled = softmax_enabled

        self._rng = np.random.default_rng(seed)
        self._current_sa_sdr = None

    @property
    def name(self):
        return 'q'

    def act(self, reward: float, state: SparseSdr, first: bool):
        if first:
            self.on_new_episode()

        s = self.sa_encoder.encode_state(state, learn=True)
        actions_sa_sdr = self.sa_encoder.encode_actions(s, learn=True)

        action_values = self.Q.values(actions_sa_sdr)
        greedy_action = np.argmax(action_values)
        if not first:
            # Q-learning step
            greedy_sa_sdr = actions_sa_sdr[greedy_action]
            self.E_traces.update(self._current_sa_sdr)
            self.Q.update(
                sa=self._current_sa_sdr, reward=reward,
                sa_next=greedy_sa_sdr,
                E_traces=self.E_traces.E
            )

        # choose action
        action = greedy_action
        if self.exploration_eps is not None and self._rng.random() < self.exploration_eps[0]:
            action = self._rng.integers(self.n_actions)
        elif self.softmax_enabled:
            action = self._rng.choice(self.n_actions, p=softmax(action_values))

        self._current_sa_sdr = actions_sa_sdr[action]
        return action

    def on_new_episode(self):
        self.Q.decay_learning_factors()
        self.E_traces.reset()
        if self.exploration_eps is not None:
            exp_decay(self.exploration_eps)


def softmax(x):
    """Computes softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def make_sa_encoder(
        env: Env, seed: int, sa_encoder_config: dict
    ):
    if sa_encoder_config:
        from htm_rl.agents.q.sa_encoders import SpSaEncoder
        return SpSaEncoder(env, seed, **sa_encoder_config)
    else:
        from htm_rl.agents.q.sa_encoders import CrossSaEncoder
        return CrossSaEncoder(env)
