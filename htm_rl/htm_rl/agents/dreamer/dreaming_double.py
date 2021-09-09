import pickle
from typing import Optional

import numpy as np
from numpy.random import Generator

from htm_rl.agents.dreamer.qvn_double import QValueNetworkDouble
from htm_rl.agents.q.eligibility_traces import EligibilityTraces
from htm_rl.agents.qmb.agent import QModelBasedAgent
from htm_rl.common.sdr import SparseSdr
from htm_rl.common.utils import multiply_decaying_value, exp_decay, isnone, softmax, DecayingValue


class DreamingDouble(QModelBasedAgent):
    Q: QValueNetworkDouble
    wake_E_traces: EligibilityTraces
    nest_traces: bool

    tm_checkpoint: Optional[bytes]

    enabled: bool
    enter_prob_alpha: DecayingValue
    enter_prob_threshold: float
    prediction_depth: int
    n_prediction_rollouts: tuple[int, int]

    starting_sa_sdr: Optional[SparseSdr]
    _episode: int

    # noinspection PyMissingConstructor
    def __init__(
            self,
            seed: int,
            wake_agent,
            enter_prob_alpha: DecayingValue,
            enter_prob_threshold: float,
            qvn: dict,
            eligibility_traces: dict,
            nest_traces: bool,
            prediction_depth: int,
            n_prediction_rollouts: tuple[int, int],
            enabled: bool = True,
            first_dreaming_episode: int = None,
            last_dreaming_episode: int = None,
            softmax_temp: DecayingValue = (0., 0.),
            exploration_eps: DecayingValue = (0., 0.),
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
        self.softmax_temp = softmax_temp
        self.exploration_eps = exploration_eps
        self.im_weight = wake_agent.im_weight
        self.ucb_estimate = wake_agent.ucb_estimate

        self.enabled = enabled
        self.enter_prob_alpha = enter_prob_alpha
        self.enter_prob_threshold = enter_prob_threshold
        self.prediction_depth = prediction_depth
        self.n_prediction_rollouts = n_prediction_rollouts
        self.first_dreaming_episode = isnone(first_dreaming_episode, 0)
        self.last_dreaming_episode = isnone(last_dreaming_episode, 999999)

        self.starting_sa_sdr = None
        self._rng = np.random.default_rng(seed)
        self._current_sa_sdr = None
        self._step = 0
        self._episode = 0

    def put_into_dream(self, starting_sa_sdr):
        self.save_tm_checkpoint()
        self.starting_sa_sdr = starting_sa_sdr.copy()

    def on_new_rollout(self, i_rollout):
        if i_rollout > 0:
            self.restore_tm_checkpoint()
        self._current_sa_sdr = self.starting_sa_sdr.copy()
        self.Q.learning_rate = multiply_decaying_value(
            self.Q.origin_learning_rate,
            self.Q.learning_rate_factor/(i_rollout + 1.)**.5
        )
        if self.E_traces.enabled:
            if self.nest_traces and self.wake_E_traces.enabled:
                self.E_traces.E = self.wake_E_traces.E.copy()
            else:
                self.E_traces.reset(decay=False)

    def wake(self):
        self.enter_prob_alpha = exp_decay(self.enter_prob_alpha)
        self.restore_tm_checkpoint()
        self.E_traces.decay_trace_decay()

    def dream(self, starting_state, prev_sa_sdr):
        self.put_into_dream(prev_sa_sdr)

        starting_state_len = len(starting_state)
        i_rollout = 0
        sum_depth = 0
        depths = []
        while i_rollout < self.n_prediction_rollouts[0] or (
                i_rollout < self.n_prediction_rollouts[1]
                and sum_depth >= 3 * i_rollout
        ):
            self.on_new_rollout(i_rollout)
            state = starting_state
            depth = 0
            for depth in range(self.prediction_depth):
                state = self.move_in_dream(state)
                if len(state) < .6 * starting_state_len:
                    break

            i_rollout += 1
            sum_depth += depth ** .8
            depths.append(depth)

        # if depths: print(depths)
        self.wake()

    def act(self, reward: float, s: SparseSdr, first: bool) -> int:
        prev_sa_sdr = self._current_sa_sdr
        actions_sa_sdr = self.sa_encoder.encode_actions(s, learn=False)

        if not first:
            # Q-learning step
            self.E_traces.update(prev_sa_sdr)
            self._make_q_learning_step(
                sa=prev_sa_sdr, r=reward, next_actions_sa_sdr=actions_sa_sdr
            )

        action = self._choose_action(actions_sa_sdr)
        chosen_sa_sdr = actions_sa_sdr[action]

        if self.ucb_estimate.enabled:
            self.ucb_estimate.update(chosen_sa_sdr)

        self._current_sa_sdr = chosen_sa_sdr
        return action

    def move_in_dream(self, state: SparseSdr):
        reward = self.reward_model.rewards[state].mean()
        _ = self.act(reward, state, first=False)

        if reward > .2:
            # found goal ==> should stop rollout
            next_s = []
            return next_s

        _, s_sa_next_superposition = self.transition_model.process(
            self._current_sa_sdr, learn=False
        )
        s_sa_next_superposition = self.transition_model.columns_from_cells(
            s_sa_next_superposition
        )
        next_s = self.sa_encoder.decode_state(s_sa_next_superposition)
        return next_s

    def can_dream(self, reward):
        if not self.enabled:
            return False
        if not (self.first_dreaming_episode <= self._episode < self.last_dreaming_episode):
            return False
        if reward > .2:
            # reward condition prevents useless planning when we've already
            # found the goal
            return False
        if self.Q.learning_rate[0] < 1e-3:
            # there's no reason to dream when learning is almost stopped
            return False
        return True

    def decide_to_dream(self, td_error):
        if self._td_error_based_dreaming(td_error):
            return True
        return False

    def _td_error_based_dreaming(self, td_error):
        max_abs_td_error = 2.
        dreaming_prob_boost = self.enter_prob_alpha[0]
        dreaming_prob = (dreaming_prob_boost * abs(td_error) - self.enter_prob_threshold)
        dreaming_prob = np.clip(dreaming_prob / max_abs_td_error, 0., 1.)
        return self._rng.random() < dreaming_prob

    def on_new_episode(self):
        self._episode += 1
        self.E_traces.decay_trace_decay()
        self.exploration_eps = exp_decay(self.exploration_eps)
        self._decay_softmax_temperature()

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
