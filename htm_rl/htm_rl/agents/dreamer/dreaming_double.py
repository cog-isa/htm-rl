import pickle
from typing import Optional

import numpy as np
from numpy.random import Generator

from htm_rl.agents.agent import Agent
from htm_rl.agents.dreamer.qvn_double import QValueNetworkDouble
from htm_rl.agents.q.agent import softmax
from htm_rl.agents.q.eligibility_traces import EligibilityTraces
from htm_rl.agents.q.sa_encoder import SaEncoder
from htm_rl.agents.qmb.reward_model import RewardModel
from htm_rl.agents.qmb.transition_model import TransitionModel
from htm_rl.agents.ucb.ucb_estimator import UcbEstimator
from htm_rl.common.sdr import SparseSdr
from htm_rl.common.utils import modify_factor_tuple, exp_decay, isnone, clip


class DreamingDouble(Agent):
    n_actions: int
    sa_encoder: SaEncoder
    Q: QValueNetworkDouble
    E_traces: EligibilityTraces
    wake_E_traces: EligibilityTraces
    nest_traces: bool

    transition_model: TransitionModel
    tm_checkpoint: Optional[bytes]
    reward_model: RewardModel

    exploration_eps: Optional[tuple[float, float]]
    softmax_enabled: bool
    im_weight: tuple[float, float]
    ucb_estimate: Optional[UcbEstimator]

    enabled: bool
    enter_prob_alpha: tuple[float, float]
    enter_prob_threshold: float
    prediction_depth: int
    n_prediction_rollouts: tuple[int, int]

    starting_sa_sdr: Optional[SparseSdr]
    dream_length: int

    _current_sa_sdr: Optional[SparseSdr]
    _rng: Generator
    _episode: int

    def __init__(
            self,
            seed: int,
            wake_agent,
            enter_prob_alpha: tuple[float, float],
            enter_prob_threshold: float,
            exploration_eps: tuple[float, float],
            qvn: dict,
            eligibility_traces: dict,
            nest_traces: bool,
            prediction_depth: int,
            n_prediction_rollouts: tuple[int, int],
            enabled: bool = True,
            first_dreaming_episode: int = 0,
            last_dreaming_episode: int = None,

    ):
        self.n_actions = wake_agent.n_actions
        self.sa_encoder = wake_agent.sa_encoder
        self.Q = QValueNetworkDouble(wake_agent.Q, **qvn)
        self.E_traces = EligibilityTraces(
            self.sa_encoder.output_sdr_size, **eligibility_traces
        )
        self.wake_E_traces = wake_agent.E_traces
        self.nest_traces = nest_traces

        self.transition_model = wake_agent.transition_model
        self.tm_checkpoint = None
        self.reward_model = wake_agent.reward_model
        self.exploration_eps = exploration_eps
        self.softmax_enabled = wake_agent.softmax_enabled
        self.im_weight = wake_agent.im_weight
        self.ucb_estimate = wake_agent.ucb_estimate

        self.enabled = enabled
        self.enter_prob_alpha = enter_prob_alpha
        self.enter_prob_threshold = enter_prob_threshold
        self.prediction_depth = prediction_depth
        self.n_prediction_rollouts = n_prediction_rollouts
        self.first_dreaming_episode = first_dreaming_episode
        self.last_dreaming_episode = isnone(last_dreaming_episode, 999999)

        self.starting_sa_sdr = None
        self.dream_length = 0

        self._rng = np.random.default_rng(seed)
        self._current_sa_sdr = None
        self._episode = 0

    def put_into_dream(self, starting_sa_sdr):
        self.save_tm_checkpoint()
        self.dream_length = 0
        self.enter_prob_alpha = exp_decay(self.enter_prob_alpha)
        self.Q.learning_rate = modify_factor_tuple(
            self.Q.origin_learning_rate, self.Q.learning_rate_factor
        )
        self.starting_sa_sdr = starting_sa_sdr.copy()

    def reset_dreaming(self, i_rollout=None):
        if i_rollout == 0:
            return

        if self.E_traces.enabled:
            if self.nest_traces and self.wake_E_traces.enabled:
                self.E_traces.E = self.wake_E_traces.E.copy()
            else:
                self.E_traces.reset(decay=False)

        if i_rollout is not None:
            self.Q.learning_rate = modify_factor_tuple(
                self.Q.origin_learning_rate,
                1.0/(i_rollout + 1.)**.5
            )
        self.restore_tm_checkpoint()
        self._current_sa_sdr = self.starting_sa_sdr.copy()

    def wake(self):
        self.reset_dreaming()

    def dream(self, starting_state, prev_sa_sdr):
        self.put_into_dream(prev_sa_sdr)

        starting_state_len = len(starting_state)
        i_rollout = 0
        sum_depth = 0
        depths = []
        while i_rollout < self.n_prediction_rollouts[0] or (
                i_rollout < self.n_prediction_rollouts[1]
                and sum_depth * self.exploration_eps[0] >= .3 * i_rollout
        ):
            self.reset_dreaming(i_rollout)
            state = starting_state
            depth = 0
            for depth in range(self.prediction_depth):
                state = self.move_in_dream(state)
                if len(state) < .6 * starting_state_len:
                    break

            sum_depth += depth ** .5
            i_rollout += 1
            depths.append(depth)

        self.dream_length += sum_depth
        # if depths:
            # print(sum_depth)
            # print(depths)
        self.wake()

    def act(self, reward: float, s: SparseSdr, first: bool) -> int:
        actions_sa_sdr = self.sa_encoder.encode_actions(s, learn=False)

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
        if self._make_random_action():
            action = self._rng.integers(self.n_actions)
        elif self.softmax_enabled:
            action = self._rng.choice(self.n_actions, p=softmax(action_values))
        elif self.ucb_estimate is not None:
            ucb_values = self.ucb_estimate.ucb_terms(actions_sa_sdr)
            action = np.argmax(action_values + ucb_values)

        self._current_sa_sdr = actions_sa_sdr[action]
        if self.ucb_estimate is not None:
            self.ucb_estimate.update(self._current_sa_sdr)

        return action

    def _make_random_action(self):
        if self.exploration_eps is not None:
            return self._rng.random() < self.exploration_eps[0]
        return False

    def move_in_dream(self, state: SparseSdr):
        reward = self.reward_model.rewards[state].mean()
        _ = self.act(reward, state, False)

        _, s_sa_next_superposition = self.transition_model.process(
            self._current_sa_sdr, learn=False
        )
        s_sa_next_superposition = self.transition_model.columns_from_cells(
            s_sa_next_superposition
        )
        next_s = self.sa_encoder.decode_state(s_sa_next_superposition)
        return next_s

    def decide_to_dream(self, td_error):
        self.dream_length = 0
        if not self.enabled:
            return False
        if not self.first_dreaming_episode <= self._episode < self.last_dreaming_episode:
            return False
        if self.Q.learning_rate[0] < 1e-3:
            return False

        if self._td_error_based_dreaming(td_error):
            return True
        return False

    def _td_error_based_dreaming(self, td_error):
        max_abs_td_error = 2.
        dreaming_prob_boost = self.enter_prob_alpha[0]
        dreaming_prob = (dreaming_prob_boost * abs(td_error) - self.enter_prob_threshold)
        dreaming_prob = clip(dreaming_prob / max_abs_td_error, 1.)
        return self._rng.random() < dreaming_prob

    def on_new_episode(self):
        if self.exploration_eps is not None:
            self.exploration_eps = exp_decay(self.exploration_eps)
        self.E_traces.reset()
        self.im_weight = exp_decay(self.im_weight)
        self._episode += 1

    def save_tm_checkpoint(self) -> bytes:
        """Saves TM state."""
        self.tm_checkpoint = pickle.dumps(self.transition_model)
        return self.tm_checkpoint

    def restore_tm_checkpoint(self, tm_checkpoint: bytes = None):
        """Restores saved TM state."""
        tm_checkpoint = isnone(tm_checkpoint, self.tm_checkpoint)
        self.transition_model = pickle.loads(tm_checkpoint)

    @property
    def name(self):
        return 'dreaming_double'
