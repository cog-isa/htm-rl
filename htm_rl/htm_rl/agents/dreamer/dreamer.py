from typing import Tuple, Optional

import numpy as np
from numpy.random import Generator

from htm_rl.agents.agent import Agent
from htm_rl.agents.dreamer.sparse_value_network import SparseValueNetwork
from htm_rl.agents.q.agent import softmax
from htm_rl.agents.q.eligibility_traces import EligibilityTraces
from htm_rl.agents.q.qvn import QValueNetwork
from htm_rl.agents.q.sa_encoder import SaEncoder
from htm_rl.agents.qmb.reward_model import RewardModel
from htm_rl.agents.qmb.transition_model import SsaTransitionModel
from htm_rl.agents.ucb.ucb_estimator import UcbEstimator
from htm_rl.common.sdr import SparseSdr
from htm_rl.common.utils import modify_factor_tuple, exp_decay, isnone, clip


class QValueNetworkDouble(QValueNetwork):
    origin_learning_rate: tuple[float, float]
    learning_rate_factor: float

    # noinspection PyMissingConstructor
    def __init__(self, origin: QValueNetwork, learning_rate_factor: float):
        self.origin_learning_rate = origin.learning_rate
        self.learning_rate_factor = learning_rate_factor
        self.learning_rate = modify_factor_tuple(origin.learning_rate, learning_rate_factor)
        self.discount_factor = origin.discount_factor
        self.cell_value = origin.cell_value
        self.last_td_error = .0


class DreamingAgent(Agent):
    n_actions: int
    sa_encoder: SaEncoder
    Q: QValueNetworkDouble
    E_traces: EligibilityTraces
    wake_E_traces: EligibilityTraces
    nest_traces: bool

    sa_transition_model: SsaTransitionModel
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

        self.sa_transition_model = wake_agent.sa_transition_model
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
        self.sa_transition_model.save_tm_state()
        self.dream_length = 0
        self.enter_prob_alpha = exp_decay(self.enter_prob_alpha)
        self.Q.learning_rate = modify_factor_tuple(self.Q.origin_learning_rate, self.Q.learning_rate_factor)
        self.starting_sa_sdr = starting_sa_sdr.copy()

    def reset_dreaming(self, i_rollout=None):
        if i_rollout == 0:
            return

        if self.E_traces.enabled:
            if self.nest_traces and self.wake_E_traces.enabled:
                self.E_traces.E = self.wake_E_traces.E.copy()
            else:
                self.E_traces.E.fill(0.)

        if i_rollout is not None:
            self.Q.learning_rate = modify_factor_tuple(
                self.Q.learning_rate,
                1.0/(i_rollout + 1.)**.5
            )
        self.sa_transition_model.restore_tm_state()
        self._current_sa_sdr = self.starting_sa_sdr.copy()

    def wake(self):
        self.reset_dreaming()

    def dream(self, starting_state, prev_sa_sdr):
        # print(self.sqvn.TD_error)
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
        #     print(depths)
        self.wake()

    def act(self, reward: float, s: SparseSdr, first: bool) -> int:
        actions_sa_sdr = self.sa_encoder.encode_actions(s, learn=False)

        action_values = self.Q.values(actions_sa_sdr)
        greedy_action = np.argmax(action_values)
        if not first:
            # Q-learning step
            greedy_sa_sdr = actions_sa_sdr[greedy_action]
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
        elif self.ucb_estimate is not None:
            ucb_values = self.ucb_estimate.ucb_terms(actions_sa_sdr)
            action = np.argmax(action_values + ucb_values)

        self._current_sa_sdr = actions_sa_sdr[action]
        if self.ucb_estimate is not None:
            self.ucb_estimate.update(self._current_sa_sdr)

        return action

    def move_in_dream(self, state: SparseSdr):
        reward = self.reward_model.rewards[state].mean()
        _ = self.act(reward, state, False)

        _, s_sa_next_superposition = self.sa_transition_model.process(state, self._current_sa_sdr, learn=False)
        s_sa_next_superposition = self.sa_transition_model.columns_from_cells(s_sa_next_superposition)
        next_s = self._split_s_sa(s_sa_next_superposition)
        return next_s

    def decide_to_dream(self):
        self.dream_length = None
        if not self.enabled:
            return False
        if not self.first_dreaming_episode <= self._episode < self.last_dreaming_episode:
            return False
        if self.Q.learning_rate[0] < 1e-3:
            return False

        td_error = self.Q.last_td_error
        max_abs_td_error = 2.
        dreaming_prob_boost = self.enter_prob_alpha[0]
        dreaming_prob = (dreaming_prob_boost * abs(td_error) - self.enter_prob_threshold)
        dreaming_prob = clip(dreaming_prob / max_abs_td_error, 1.)
        return self._rng.random() < dreaming_prob

    def _split_s_sa(self, s_a: SparseSdr, with_options=False):
        size = self.sa_encoder.state_sp.output_sdr_size
        state = np.array([x for x in s_a if x < size])
        if not with_options:
            return state

        # FIXME
        actions = [x - size for x in s_a if x >= size]
        actions_activation = [0 for _ in range(self.n_actions)]
        for x in actions:
            actions_activation[self.sa_encoder.action_encoder.decode_bit(x)] += 1

        actions = []
        for action, activation in enumerate(actions_activation):
            if self.sa_encoder.action_encoder.activation_fraction(activation) > .3:
                actions.append(action)

        return state, actions

    def on_new_episode(self):
        if self.exploration_eps is not None:
            self.exploration_eps = exp_decay(self.exploration_eps)
        self.im_weight = exp_decay(self.im_weight)
        self._episode += 1

    @property
    def name(self):
        return 'fake: dreaming'


class DreamerOld:
    svn: SparseValueNetwork

    enabled: bool
    dreaming_prob_alpha: Tuple[float, float]
    dreaming_prob_threshold: float
    rnd_move_prob: float
    learning_rate: Tuple[float, float]
    learning_rate_factor: float
    td_lambda: bool
    trace_decay: float
    nest_traces: bool
    cell_eligibility_trace: Optional[np.ndarray]
    starting_sa_sdr: Optional[SparseSdr]
    TD_error: Optional[float]

    def __init__(
            self, svn: SparseValueNetwork,
            dreaming_prob_alpha: Tuple[float, float], dreaming_prob_threshold: float,
            rnd_move_prob: float,
            learning_rate_factor: float, trace_decay: Optional[float],
            enabled: bool = True, nest_traces: bool = True
    ):
        if trace_decay is None:
            trace_decay = svn.trace_decay

        self.enabled = enabled
        self.dreaming_prob_alpha = dreaming_prob_alpha
        self.dreaming_prob_threshold = dreaming_prob_threshold
        self.rnd_move_prob = rnd_move_prob
        self.learning_rate_factor = learning_rate_factor
        self.learning_rate = svn.learning_rate
        self.td_lambda = trace_decay > .0
        self.trace_decay = trace_decay
        self.nest_traces = nest_traces

        self.svn = svn
        self.cell_eligibility_trace = None
        self.starting_sa_sdr = None
        self.TD_error = None

    def put_into_dream(self, starting_sa_sdr):
        wake_svn = self.svn
        self.learning_rate = wake_svn.learning_rate
        wake_svn.learning_rate = modify_factor_tuple(wake_svn.learning_rate, self.learning_rate_factor)
        wake_svn.trace_decay, self.trace_decay = self.trace_decay, wake_svn.trace_decay

        self.cell_eligibility_trace = wake_svn.cell_eligibility_trace.copy()
        if not self.nest_traces:
            wake_svn.cell_eligibility_trace.fill(.0)

        self.starting_sa_sdr = starting_sa_sdr.copy()
        self.TD_error = wake_svn.TD_error

    def reset_dreaming(self, i_rollout=None):
        dreaming_svn = self.svn
        dreaming_svn.cell_eligibility_trace = self.cell_eligibility_trace.copy()
        if not self.nest_traces:
            dreaming_svn.cell_eligibility_trace.fill(.0)
        if i_rollout is not None:
            dreaming_svn.learning_rate = modify_factor_tuple(
                dreaming_svn.learning_rate,
                1.0/(i_rollout + 1.)**.5
            )
        return self.starting_sa_sdr.copy()

    def wake(self):
        dreaming_svn = self.svn
        dreaming_svn.learning_rate = self.learning_rate
        dreaming_svn.trace_decay, self.trace_decay = self.trace_decay, dreaming_svn.trace_decay
        dreaming_svn.TD_error = self.TD_error
        dreaming_svn.cell_eligibility_trace = self.cell_eligibility_trace.copy()