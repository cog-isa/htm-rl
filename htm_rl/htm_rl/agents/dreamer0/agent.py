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

    train: bool
    im_weight: tuple[float, float]
    exploration_eps: Optional[tuple[float, float]]
    softmax_enabled: bool

    td_error_decay: float
    cum_td_error: float
    force_dreaming: bool
    dreamer: DreamingDouble

    prediction_depth: int
    n_prediction_rollouts: Tuple[int, int]
    dream_length: Optional[int]

    _step: int
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

        self.train = True
        self.exploration_eps = exploration_eps
        self.softmax_enabled = softmax_enabled
        self.im_weight = im_weight
        self.ucb_estimate = None
        if ucb_estimate:
            self.ucb_estimate = UcbEstimator(self.sa_encoder.output_sdr_size, **ucb_estimate)

        self._rng = np.random.default_rng(seed)
        self._current_sa_sdr = None
        self._step = 0

        self.td_error_decay = td_error_decay
        self.cum_td_error = 0.
        self.force_dreaming = False
        self.dreamer = DreamingDouble(seed, self, **dreaming)

    @property
    def name(self):
        return 'dreamer'

    def on_new_episode(self):
        self._step = 0
        self.E_traces.reset()
        if self.train:
            self.Q.decay_learning_factors()
            if self.exploration_eps is not None:
                self.exploration_eps = exp_decay(self.exploration_eps)
            self.reward_model.decay_learning_factors()
            self.im_weight = exp_decay(self.im_weight)
            if self.ucb_estimate is not None:
                self.ucb_estimate.decay_learning_factors()
            self.transition_model.reset()
            self.dreamer.on_new_episode()

    def act(self, reward: float, state: SparseSdr, first: bool):
        if first and self._step > 0:
            self.on_new_episode()
            return None

        train = self.train
        im_reward = 0
        if train:
            x = (1 - self.transition_model.recall) ** 2
            im_reward = self.im_weight[0] * x

        prev_sa_sdr = self._current_sa_sdr
        s = self.sa_encoder.encode_state(state, learn=True and train)
        actions_sa_sdr = self.sa_encoder.encode_actions(s, learn=True and train)

        if train and not first:
            self.reward_model.update(s, reward)
            self.E_traces.update(prev_sa_sdr)
            self._make_q_learning_step(
                sa=prev_sa_sdr, r=reward+im_reward,
                next_actions_sa_sdr=actions_sa_sdr
            )
            self._try_dreaming(
                prev_sa=prev_sa_sdr, r=reward+im_reward,
                s=s, next_actions_sa_sdr=actions_sa_sdr, eval_r=reward
            )

        action = self._choose_action(actions_sa_sdr)
        chosen_sa_sdr = actions_sa_sdr[action]
        if train:
            self.transition_model.process(
                self.transition_model.preprocess(s, chosen_sa_sdr),
                learn=True and train
            )
        if train and self.ucb_estimate is not None:
            self.ucb_estimate.update(chosen_sa_sdr)

        self._current_sa_sdr = actions_sa_sdr[action]
        self._step += 1
        return action

    def _choose_action(self, next_actions_sa_sdr):
        if self.train and self._make_random_action():
            return self._rng.integers(self.n_actions)
        if self.softmax_enabled:
            action_values = self.Q.values(next_actions_sa_sdr)
            return self._rng.choice(self.n_actions, p=softmax(action_values))
        if self.train and self.ucb_estimate is not None:
            action_values = self.Q.values(next_actions_sa_sdr)
            ucb_values = self.ucb_estimate.ucb_terms(next_actions_sa_sdr)
            return np.argmax(action_values + ucb_values)

        action_values = self.Q.values(next_actions_sa_sdr)
        greedy_action = np.argmax(action_values)
        return greedy_action

    def _make_q_learning_step(self, sa, r, next_actions_sa_sdr):
        action_values = self.Q.values(next_actions_sa_sdr)
        greedy_action = np.argmax(action_values)
        greedy_sa_sdr = next_actions_sa_sdr[greedy_action]
        self.Q.update(
            sa=sa, reward=r, sa_next=greedy_sa_sdr,
            E_traces=self.E_traces.E
        )

    def _try_dreaming(self, prev_sa, r, s, next_actions_sa_sdr, eval_r):
        if eval_r > 0.1:
            # prevents highly likeable useless planning in the end
            return

        if not self.force_dreaming:
            action_values = self.Q.values(next_actions_sa_sdr)
            greedy_action = np.argmax(action_values)
            greedy_sa_sdr = next_actions_sa_sdr[greedy_action]
            td_error = self.Q.td_error(prev_sa, r, greedy_sa_sdr)

            self.cum_td_error = exp_sum(self.cum_td_error, self.td_error_decay, td_error)
            dream = self.dreamer.decide_to_dream(self.cum_td_error)
        else:
            dream = True
            self.force_dreaming = False

        if dream:
            # print('-- dream --')
            self.dreamer.dream(s, prev_sa)

    def _make_random_action(self):
        if self.exploration_eps is not None:
            return self._rng.random() < self.exploration_eps[0]
        return False
