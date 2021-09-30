from typing import Optional

import numpy as np
from numpy.random import Generator

from htm_rl.agents.agent import Agent
from htm_rl.agents.q.eligibility_traces import EligibilityTraces
from htm_rl.agents.q.qvn import QValueNetwork
from htm_rl.agents.q.sa_encoder import SaEncoder
from htm_rl.agents.q.sa_encoders import make_sa_encoder
from htm_rl.agents.q.ucb_estimator import UcbEstimator
from htm_rl.common.sdr import SparseSdr
from htm_rl.common.utils import exp_decay, softmax, DecayingValue, isnone
from htm_rl.envs.env import Env


class QAgent(Agent):
    n_actions: int
    sa_encoder: SaEncoder
    Q: QValueNetwork
    E_traces: EligibilityTraces

    train: bool
    softmax_temp: DecayingValue
    softmax_limit = .04
    exploration_eps: DecayingValue
    ucb_estimate: UcbEstimator

    _step: int
    _current_sa_sdr: Optional[SparseSdr]
    _rng: Generator

    def __init__(
            self,
            env: Env,
            seed: int,
            qvn: dict,
            sa_encoder: dict = None,
            eligibility_traces: dict = None,
            softmax_temp: DecayingValue = (0., 0.),
            exploration_eps: DecayingValue = (0., 0.),
            ucb_estimate: dict = None,
    ):
        self.n_actions = env.n_actions
        self.sa_encoder = make_sa_encoder(env, seed, sa_encoder)
        self.Q = QValueNetwork(self.sa_encoder.output_sdr_size, seed, **qvn)
        self.E_traces = EligibilityTraces(
            self.sa_encoder.output_sdr_size,
            **isnone(eligibility_traces, {})
        )

        self.train = True
        self.softmax_temp = softmax_temp
        self.exploration_eps = exploration_eps
        self.ucb_estimate = UcbEstimator(
            self.sa_encoder.output_sdr_size, **isnone(ucb_estimate, {})
        )

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
            self._decay_softmax_temperature()
            self.exploration_eps = exp_decay(self.exploration_eps)
            if self.ucb_estimate.enabled:
                self.ucb_estimate.decay_learning_factors()

    def act(self, reward: float, state: SparseSdr, first: bool) -> Optional[int]:
        if first and self._step > 0:
            self.on_new_episode()
            return None

        train = self.train
        prev_sa_sdr = self._current_sa_sdr
        s = self.sa_encoder.encode_state(state, learn=True and train)
        actions_sa_sdr = self._encode_s_actions(s, learn=True and train)

        if train and not first:
            self.E_traces.update(prev_sa_sdr)
            self._make_q_learning_step(
                sa=prev_sa_sdr, r=reward, next_actions_sa_sdr=actions_sa_sdr
            )

        action = self._choose_action(actions_sa_sdr)
        chosen_sa_sdr = actions_sa_sdr[action]

        if train and self.ucb_estimate.enabled:
            self.ucb_estimate.update(chosen_sa_sdr)

        self._current_sa_sdr = chosen_sa_sdr
        self._step += 1
        return action

    def _choose_action(self, next_actions_sa_sdr: list[SparseSdr]) -> int:
        if self.softmax_temp[0] >= self.softmax_limit:
            # SOFTMAX
            action_values = self.Q.values(next_actions_sa_sdr)
            p = softmax(action_values, self.softmax_temp[0])
            return self._rng.choice(self.n_actions, p=p)

        if self.train and self._should_make_random_action():
            # RND
            return self._rng.integers(self.n_actions)

        if self.train and self.ucb_estimate.enabled:
            # UCB
            action_values = self.Q.values(next_actions_sa_sdr)
            ucb_values = self.ucb_estimate.ucb_terms(next_actions_sa_sdr)
            # noinspection PyTypeChecker
            return np.argmax(action_values + ucb_values)

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
        if self.exploration_eps[0] < .001:
            # === disabled
            return False
        return self._rng.random() < self.exploration_eps[0]

    def _decay_softmax_temperature(self):
        if self.softmax_temp[0] == 0.:
            return

        temp, decay = exp_decay(self.softmax_temp)
        # limit the "hardness"
        temp = max(temp, self.softmax_limit)
        self.softmax_temp = temp, decay

    def _encode_s_actions(self, s: SparseSdr, learn: bool) -> list[SparseSdr]:
        p_learn = 1.5 / self.n_actions

        return [
            self.sa_encoder.encode_s_action(
                s, action,
                learn=learn and self._rng.random() < p_learn
            )
            for action in range(self.n_actions)
        ]
