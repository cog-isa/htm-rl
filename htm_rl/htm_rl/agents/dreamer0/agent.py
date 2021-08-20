from typing import Tuple, Optional

import numpy as np
from numpy.random import Generator

from htm_rl.agents.agent import Agent
from htm_rl.agents.dreamer.dreaming_double import DreamingDouble
from htm_rl.agents.q.agent import softmax, make_sa_encoder
from htm_rl.agents.q.eligibility_traces import EligibilityTraces
from htm_rl.agents.q.qvn import QValueNetwork
from htm_rl.agents.q.sa_encoder import SaEncoder
from htm_rl.agents.qmb.agent import make_transition_model
from htm_rl.agents.qmb.reward_model import RewardModel
from htm_rl.agents.qmb.transition_model import TransitionModel
from htm_rl.agents.ucb.ucb_estimator import UcbEstimator
from htm_rl.common.sdr import SparseSdr
from htm_rl.common.utils import exp_decay, exp_sum
from htm_rl.envs.env import Env


class DreamerAgent(Agent):
    n_actions: int
    sa_encoder: SaEncoder
    Q: QValueNetwork
    E_traces: Optional[EligibilityTraces]

    transition_model: TransitionModel
    reward_model: RewardModel

    im_weight: tuple[float, float]
    exploration_eps: Optional[tuple[float, float]]
    softmax_enabled: bool

    td_error_decay: float
    cum_td_error: float
    dreamer: DreamingDouble

    prediction_depth: int
    n_prediction_rollouts: Tuple[int, int]
    dream_length: Optional[int]

    _current_sa_sdr: Optional[SparseSdr]
    _rng: Generator

    def __init__(
            self,
            env: Env,
            seed: int,
            qvn: dict,
            reward_model: dict,
            transition_model: dict,
            im_weight: tuple[float, float],
            td_error_decay: float,
            dreaming: dict,
            eligibility_traces: dict,
            exploration_eps: tuple[float, float] = None,
            softmax_enabled: bool = False,
            ucb_estimate: dict = None,
            sa_encoder: dict = None,
    ):
        self.n_actions = env.n_actions
        self.sa_encoder = make_sa_encoder(env, seed, sa_encoder)
        self.Q = QValueNetwork(self.sa_encoder.output_sdr_size, seed, **qvn)
        self.E_traces = EligibilityTraces(
            self.sa_encoder.output_sdr_size,
            **eligibility_traces
        )
        self.transition_model = make_transition_model(
            self.sa_encoder, transition_model
        )
        self.reward_model = RewardModel(self.sa_encoder.output_sdr_size, **reward_model)
        self.exploration_eps = exploration_eps
        self.softmax_enabled = softmax_enabled
        self.im_weight = im_weight
        self.ucb_estimate = None
        if ucb_estimate:
            self.ucb_estimate = UcbEstimator(self.sa_encoder.output_sdr_size, **ucb_estimate)

        self._rng = np.random.default_rng(seed)
        self._current_sa_sdr = None
        self._episode = 0

        self.td_error_decay = td_error_decay
        self.cum_td_error = 0.
        self.dreamer = DreamingDouble(seed, self, **dreaming)

    @property
    def name(self):
        return 'dreamer'

    def act(self, reward: float, state: SparseSdr, first: bool):
        if first:
            self.on_new_episode()

        print(f'S: {state}')

        x = (1 - self.transition_model.recall) ** 2
        im_reward = self.im_weight[0] * x

        s = self.sa_encoder.encode_state(state, learn=True)
        actions_sa_sdr = self.sa_encoder.encode_actions(s, learn=True)

        action_values = self.Q.values(actions_sa_sdr)
        greedy_action = np.argmax(action_values)
        if not first:
            self.reward_model.update(s, reward)
            # Q-learning step
            greedy_sa_sdr = actions_sa_sdr[greedy_action]
            self.Q.update(
                sa=self._current_sa_sdr, reward=reward + im_reward,
                sa_next=greedy_sa_sdr,
                E_traces=self.E_traces.E
            )
            self.cum_td_error = exp_sum(self.cum_td_error, self.td_error_decay, self.Q.last_td_error)

        if not first and reward <= 0.1 and self.dreamer.decide_to_dream(self.cum_td_error):
            # condition prevents inevitable useless planning in the end
            self.dreamer.dream(s, self._current_sa_sdr)
            action_values = self.Q.values(actions_sa_sdr)
            greedy_action = np.argmax(action_values)

        # choose action
        action = greedy_action
        if self.exploration_eps is not None and self._rng.random() < self.exploration_eps[0]:
            action = self._rng.integers(self.n_actions)
        elif self.softmax_enabled:
            action = self._rng.choice(self.n_actions, p=softmax(action_values))
        elif self.ucb_estimate is not None:
            ucb_values = self.ucb_estimate.ucb_terms(actions_sa_sdr)
            action = np.argmax(action_values + ucb_values)

        self._current_sa_sdr = actions_sa_sdr[action]
        self.transition_model.process(
            self.transition_model.preprocess(s, self._current_sa_sdr),
            learn=True
        )
        if self.ucb_estimate is not None:
            self.ucb_estimate.update(self._current_sa_sdr)
        print(f'A: {self._current_sa_sdr}')
        return action

    def on_new_episode(self):
        self.Q.decay_learning_factors()
        self.E_traces.reset()
        if self.exploration_eps is not None:
            self.exploration_eps = exp_decay(self.exploration_eps)
        self.transition_model.reset()
        self.reward_model.decay_learning_factors()
        self.im_weight = exp_decay(self.im_weight)
        if self.ucb_estimate is not None:
            self.ucb_estimate.decay_learning_factors()
        self.dreamer.on_new_episode()
