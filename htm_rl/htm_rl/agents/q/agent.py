from typing import Optional

import numpy as np
from numpy.random import Generator

from htm_rl.agents.agent import Agent
from htm_rl.agents.q.eligibility_traces import EligibilityTraces
from htm_rl.agents.q.qvn import QValueNetwork
from htm_rl.agents.q.sa_encoder import SaEncoder
from htm_rl.agents.q.sa_encoders import make_sa_encoder
from htm_rl.agents.ucb.ucb_estimator import UcbEstimator
from htm_rl.common.sdr import SparseSdr
from htm_rl.common.utils import exp_decay, softmax, DecayingValue
from htm_rl.envs.env import Env


class QAgent(Agent):
    n_actions: int
    sa_encoder: SaEncoder
    Q: QValueNetwork
    E_traces: EligibilityTraces

    train: bool
    exploration_eps: DecayingValue
    ucb_estimate: Optional[UcbEstimator]
    softmax_enabled: bool

    _step: int
    _current_sa_sdr: Optional[SparseSdr]
    _rng: Generator

    def __init__(
            self,
            env: Env,
            seed: int,
            qvn: dict,
            eligibility_traces: dict,
            exploration_eps: DecayingValue,
            sa_encoder: dict = None,
            ucb_estimate: dict = None,
            softmax_enabled: bool = False,
    ):
        self.n_actions = env.n_actions
        self.sa_encoder = make_sa_encoder(env, seed, sa_encoder)
        self.Q = QValueNetwork(self.sa_encoder.output_sdr_size, seed, **qvn)
        self.E_traces = EligibilityTraces(
            self.sa_encoder.output_sdr_size,
            **eligibility_traces
        )

        self.train = True
        self.exploration_eps = exploration_eps
        self.softmax_enabled = softmax_enabled
        self.ucb_estimate = None
        if ucb_estimate:
            self.ucb_estimate = UcbEstimator(self.sa_encoder.output_sdr_size, **ucb_estimate)

        self._rng = np.random.default_rng(seed)
        self._current_sa_sdr = None
        self._step = 0

    @property
    def name(self):
        return 'q'

    def on_new_episode(self):
        self._step = 0
        self.E_traces.reset()
        if self.train:
            self.Q.decay_learning_factors()
            self.exploration_eps = exp_decay(self.exploration_eps)
            if self.ucb_estimate is not None:
                self.ucb_estimate.decay_learning_factors()

    def act(self, reward: float, state: SparseSdr, first: bool) -> Optional[int]:
        if first and self._step > 0:
            self.on_new_episode()
            return None

        train = self.train
        prev_sa_sdr = self._current_sa_sdr
        s = self.sa_encoder.encode_state(state, learn=True and train)
        actions_sa_sdr = self.sa_encoder.encode_actions(s, learn=True and train)

        if train and not first:
            self.E_traces.update(prev_sa_sdr)
            self._make_q_learning_step(
                sa=prev_sa_sdr, r=reward, next_actions_sa_sdr=actions_sa_sdr
            )

        action = self._choose_action(actions_sa_sdr)
        chosen_sa_sdr = actions_sa_sdr[action]

        if train and self.ucb_estimate is not None:
            self.ucb_estimate.update(chosen_sa_sdr)

        self._current_sa_sdr = chosen_sa_sdr
        self._step += 1
        return action

    def _choose_action(self, next_actions_sa_sdr: list[SparseSdr]) -> int:
        if self.train and self._should_make_random_action():
            # RND
            return self._rng.integers(self.n_actions)

        if self.train and self.ucb_estimate is not None:
            # UCB
            action_values = self.Q.values(next_actions_sa_sdr)
            ucb_values = self.ucb_estimate.ucb_terms(next_actions_sa_sdr)
            # noinspection PyTypeChecker
            return np.argmax(action_values + ucb_values)

        if self.softmax_enabled:
            # SOFTMAX
            action_values = self.Q.values(next_actions_sa_sdr)
            return self._rng.choice(self.n_actions, p=softmax(action_values))

        # GREEDY
        action_values = self.Q.values(next_actions_sa_sdr)
        greedy_action = np.argmax(action_values)
        # noinspection PyTypeChecker
        return greedy_action

    def _make_q_learning_step(
            self, sa: SparseSdr, r: float, next_actions_sa_sdr: list[SparseSdr]
    ):
        action_values = self.Q.values(next_actions_sa_sdr)
        greedy_action = np.argmax(action_values)
        greedy_sa_sdr = next_actions_sa_sdr[greedy_action]
        self.Q.update(
            sa=sa, reward=r, sa_next=greedy_sa_sdr,
            E_traces=self.E_traces.E
        )

    def _should_make_random_action(self) -> bool:
        if self.exploration_eps is None:
            return False
        return self._rng.random() < self.exploration_eps[0]
