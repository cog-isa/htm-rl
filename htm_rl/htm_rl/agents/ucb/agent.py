from typing import Optional

import numpy as np

from htm_rl.agents.agent import Agent
from htm_rl.agents.q.eligibility_traces import EligibilityTraces
from htm_rl.agents.q.qvn import QValueNetwork
from htm_rl.agents.q.sa_encoder import SaEncoder
from htm_rl.agents.ucb.ucb_estimator import UcbEstimator
from htm_rl.common.sdr import SparseSdr
from htm_rl.envs.env import Env


class UcbAgent(Agent):
    n_actions: int
    sa_encoder: SaEncoder
    Q: QValueNetwork
    E_traces: Optional[EligibilityTraces]
    ucb_estimate: UcbEstimator

    _current_sa_sdr: Optional[SparseSdr]

    def __init__(
            self,
            env: Env,
            seed: int,
            sa_encoder: dict,
            qvn: dict,
            ucb_estimate: dict,
            eligibility_traces: dict = None,
    ):
        self.n_actions = env.n_actions
        self.sa_encoder = SaEncoder(env, seed, **sa_encoder)
        self.Q = QValueNetwork(self.sa_encoder.output_sdr_size, seed, **qvn)
        self.E_traces = EligibilityTraces(
            self.sa_encoder.output_sdr_size,
            **eligibility_traces
        )
        self.ucb_estimate = UcbEstimator(self.sa_encoder.output_sdr_size, **ucb_estimate)

    @property
    def name(self):
        return 'ucb'

    def act(self, reward: float, state: SparseSdr, first: bool):
        if first:
            self.on_new_episode()

        s = self.sa_encoder.encode_state(state, learn=True)
        actions_sa_sdr = self.sa_encoder.encode_actions(s, learn=True)

        action_values = self.Q.values(actions_sa_sdr)
        if not first:
            # Q-learning step
            greedy_action = np.argmax(action_values)
            greedy_sa_sdr = actions_sa_sdr[greedy_action]
            self.Q.update(
                sa=self._current_sa_sdr, reward=reward,
                sa_next=greedy_sa_sdr,
                E_traces=self.E_traces.E
            )

        # choose action
        ucb_values = self.ucb_estimate.ucb_terms(actions_sa_sdr)
        action = np.argmax(action_values + ucb_values)

        self._current_sa_sdr = actions_sa_sdr[action]
        self.ucb_estimate.update(self._current_sa_sdr)
        return action

    def on_new_episode(self):
        self.Q.decay_learning_factors()
        self.E_traces.reset()
        self.ucb_estimate.decay_learning_factors()
