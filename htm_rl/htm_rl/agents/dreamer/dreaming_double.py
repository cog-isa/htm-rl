import pickle
from typing import Optional

import numpy as np
from numpy.random import Generator

from htm_rl.agents.dreamer.falling_asleep import TdErrorBasedFallingAsleep, AnomalyBasedFallingAsleep
from htm_rl.agents.dreamer.qvn_double import QValueNetworkDouble
from htm_rl.agents.q.eligibility_traces import EligibilityTraces
from htm_rl.agents.qmb.agent import QModelBasedAgent
from htm_rl.common.sdr import SparseSdr
from htm_rl.common.utils import multiply_decaying_value, exp_decay, isnone, DecayingValue


class DreamingDouble(QModelBasedAgent):
    Q: QValueNetworkDouble
    wake_E_traces: EligibilityTraces
    derive_E_traces: bool
    rollout_q_lr_decay_power: float
    trajectory_exploration_eps_decay: float

    tm_checkpoint: Optional[bytes]

    enabled: bool
    falling_asleep_strategy: str
    td_error_based_falling_asleep: Optional[TdErrorBasedFallingAsleep]
    anomaly_based_falling_asleep: Optional[AnomalyBasedFallingAsleep]
    prediction_depth: int
    n_prediction_rollouts: tuple[int, int]

    _starting_sa_sdr: Optional[SparseSdr]
    _starting_state: Optional[SparseSdr]
    _exploration_eps_backup: Optional[DecayingValue]
    _episode: int

    # noinspection PyMissingConstructor,PyPep8Naming
    def __init__(
            self,
            seed: int,
            wake_agent: QModelBasedAgent,
            falling_asleep_strategy: str,
            qvn: dict,
            eligibility_traces: dict,
            derive_e_traces: bool,
            prediction_depth: int,
            n_prediction_rollouts: tuple[int, int],
            enabled: bool = True,
            first_dreaming_episode: int = None,
            last_dreaming_episode: int = None,
            rollout_q_lr_decay_power: float = 0.,
            softmax_temp: DecayingValue = (0., 0.),
            exploration_eps: DecayingValue = (0., 0.),
            trajectory_exploration_eps_decay: float = 1.,
            td_error_based_falling_asleep: dict = None,
            anomaly_based_falling_asleep: dict = None,
    ):
        self.n_actions = wake_agent.n_actions
        self.sa_encoder = wake_agent.sa_encoder
        self.Q = QValueNetworkDouble(wake_agent.Q, **qvn)
        self.rollout_q_lr_decay_power = rollout_q_lr_decay_power
        self.E_traces = EligibilityTraces(
            self.sa_encoder.output_sdr_size, **eligibility_traces
        )
        self.wake_E_traces = wake_agent.E_traces
        self.derive_E_traces = derive_e_traces

        self.transition_model = wake_agent.transition_model
        self.tm_checkpoint = None
        self.anomaly_model = wake_agent.anomaly_model
        self.reward_model = wake_agent.reward_model

        self.softmax_temp = softmax_temp
        self.exploration_eps = exploration_eps
        self.trajectory_exploration_eps_decay = trajectory_exploration_eps_decay
        self.im_weight = wake_agent.im_weight
        self.ucb_estimate = wake_agent.ucb_estimate

        self.enabled = enabled
        self.falling_asleep_strategy = falling_asleep_strategy
        self.td_error_based_falling_asleep = TdErrorBasedFallingAsleep(
            **td_error_based_falling_asleep
        )
        self.anomaly_based_falling_asleep = AnomalyBasedFallingAsleep(
            **anomaly_based_falling_asleep
        )
        self.prediction_depth = prediction_depth
        self.n_prediction_rollouts = n_prediction_rollouts
        self.first_dreaming_episode = isnone(first_dreaming_episode, 0)
        self.last_dreaming_episode = isnone(last_dreaming_episode, 999999)

        self.train = True
        self._starting_sa_sdr = None
        self._starting_state = None
        self._exploration_eps_backup = None
        self._rng = np.random.default_rng(seed)
        self._current_sa_sdr = None
        self._step = 0
        self._episode = 0

    def on_new_episode(self):
        self._episode += 1
        self.td_error_based_falling_asleep.boost_prob_alpha = exp_decay(
            self.td_error_based_falling_asleep.boost_prob_alpha
        )
        self.E_traces.decay_trace_decay()
        self.exploration_eps = exp_decay(self.exploration_eps)
        self._decay_softmax_temperature()

    def can_dream(self, reward):
        if not self.enabled:
            return False
        if not (self.first_dreaming_episode <= self._episode < self.last_dreaming_episode):
            return False
        if reward > .2:
            # reward condition prevents useless planning when we've already
            # found the goal
            return False
        if self.Q.learning_rate[0] < 1e-4:
            # there's no reason to dream when learning is almost stopped
            return False
        return True

    def decide_to_dream(self, td_error: float = None, anomaly: float = None):
        strategy = self.falling_asleep_strategy
        if strategy == 'td_error':
            return self._td_error_based_dreaming(td_error)
        elif strategy == 'anomaly':
            return self._anomaly_based_dreaming(anomaly)
        return False

    def dream(self, starting_state, prev_sa_sdr):
        self._put_into_dream(starting_state, prev_sa_sdr)

        # from htm_rl.agents.q.debug.state_encoding_provider import StateEncodingProvider
        # s_enc_provider: StateEncodingProvider = self.s_enc_provider
        # s_enc_provider.reset()
        # s_enc_provider.get_encoding_scheme()
        # print('pos', self.env.agent.position)
        #
        # sa_encoder: SpSaEncoder = self.sa_encoder
        # if sa_encoder.state_clusters:
        #     i_cluster, sims, cluster = sa_encoder.match_nearest_cluster(starting_state)
        #     print('cl', cluster, i_cluster, sims)

        starting_state_len = len(starting_state)
        i_rollout = 0
        sum_depth = 0
        depths = []
        while i_rollout < self.n_prediction_rollouts[0] or (
                i_rollout < self.n_prediction_rollouts[1]
                and 2. <= sum_depth < 3.2
        ):
            self._on_new_rollout(i_rollout)
            state = starting_state
            depth = 0
            for depth in range(self.prediction_depth):
                next_state, a, anomaly = self._move_in_dream(state)
                if len(next_state) < .6 * starting_state_len or anomaly > .7:
                    break
                state = next_state

            i_rollout += 1
            sum_depth += depth ** .6
            depths.append(depth)

        # if depths: print(depths)
        self._wake()

    def _move_in_dream(self, state: SparseSdr):
        # from htm_rl.agents.q.debug.state_encoding_provider import StateEncodingProvider
        # s_enc_provider: StateEncodingProvider = self.s_enc_provider
        # scheme = s_enc_provider.get_encoding_scheme()
        #
        # position, _, _ = s_enc_provider.decode_state(state, scheme, .0)
        # print('s pos', state, position)

        s = state
        reward = self.reward_model.state_reward(s)
        action = self.act(reward, s, first=False)

        # from htm_rl.envs.biogwlab.move_dynamics import DIRECTIONS_ORDER
        # print('r, a', reward, action, DIRECTIONS_ORDER[action])

        if reward > .3:
            # found goal ==> should stop rollout
            return [], action, 1.

        self.transition_model.process(s, learn=False)
        s_a = self.sa_encoder.concat_s_action(s, action, learn=False)
        # print('s_a', s_a)
        self.transition_model.process(s_a, learn=False)
        s_next = self.transition_model.predicted_cols

        allowed_max_len = int(1.2 * len(self._starting_state))
        if len(s_next) > allowed_max_len:
            s_next = self._rng.choice(s_next, allowed_max_len, replace=False)
            s_next.sort()

        # s_next = self.sa_encoder.restore_s(s_next, .45)
        # print('s_next', s_next)

        # position, overlap, s_ = s_enc_provider.decode_state(s_next, scheme, .4)
        # print(position, overlap, s_)
        # print('====================')
        # if s_ is not None:
        #     s_next = np.array(list(sorted(s_)))

        anomaly = self.anomaly_model.state_anomaly(s_next, prev_action=action)
        return s_next, action, anomaly

    def _put_into_dream(self, starting_state: SparseSdr, starting_sa_sdr: SparseSdr):
        self._save_tm_checkpoint()
        self._starting_state = starting_state.copy()
        self._starting_sa_sdr = starting_sa_sdr.copy()
        self._exploration_eps_backup = self.exploration_eps

    def _on_new_rollout(self, i_rollout):
        if i_rollout > 0:
            self._restore_tm_checkpoint()
        self._current_sa_sdr = self._starting_sa_sdr.copy()
        # noinspection PyPep8Naming
        Q_lr_decay = 1. / (i_rollout + 1.)**self.rollout_q_lr_decay_power
        self.Q.learning_rate = multiply_decaying_value(
            self.Q.origin_learning_rate,
            self.Q.learning_rate_factor * Q_lr_decay
        )
        if self.E_traces.enabled:
            if self.derive_E_traces and self.wake_E_traces.enabled:
                self.E_traces.E = self.wake_E_traces.E.copy()
            else:
                self.E_traces.reset(decay=False)

    def _wake(self):
        self.exploration_eps = self._exploration_eps_backup
        self._restore_tm_checkpoint()

    def act(self, reward: float, s: SparseSdr, first: bool) -> int:
        prev_sa_sdr = self._current_sa_sdr
        actions_sa_sdr = self._encode_s_actions(s, learn=False)

        if not first:
            # Q-learning step
            self.E_traces.update(prev_sa_sdr, with_reset=False)
            self._make_q_learning_step(
                sa=prev_sa_sdr, r=reward, next_actions_sa_sdr=actions_sa_sdr
            )

        action = self._choose_action(actions_sa_sdr)
        chosen_sa_sdr = actions_sa_sdr[action]

        # reduce eps-based exploration on later dreaming trajectory steps
        self.exploration_eps = multiply_decaying_value(
            self.exploration_eps, self.trajectory_exploration_eps_decay
        )

        if self.ucb_estimate.enabled:
            self.ucb_estimate.update(chosen_sa_sdr)

        self._current_sa_sdr = chosen_sa_sdr
        return action

    def _td_error_based_dreaming(self, td_error):
        boost_prob_alpha = self.td_error_based_falling_asleep.boost_prob_alpha[0]
        prob_threshold = self.td_error_based_falling_asleep.prob_threshold
        max_abs_td_error = 2.
        p = (boost_prob_alpha * abs(td_error) - prob_threshold)
        p = np.clip(p / max_abs_td_error, 0., 1.)
        return self._rng.random() < p

    def _anomaly_based_dreaming(self, anomaly):
        params = self.anomaly_based_falling_asleep

        if anomaly > params.anomaly_threshold:
            return False

        p = 1 - anomaly

        # --> [0, >1] --> [0, 1]
        alpha, beta = params.alpha, params.beta
        p = (p / alpha)**beta
        p /= (1 / alpha)**beta

        # --> [0, max_prob]
        p *= params.max_prob        # [0, max_prob]
        return self._rng.random() < p

    def _save_tm_checkpoint(self) -> bytes:
        """Saves TM state."""
        if self.transition_model.tm.cells_per_column > 1:
            self.tm_checkpoint = pickle.dumps(self.transition_model)
            return self.tm_checkpoint

    def _restore_tm_checkpoint(self, tm_checkpoint: bytes = None):
        """Restores saved TM state."""
        if self.transition_model.tm.cells_per_column > 1:
            tm_checkpoint = isnone(tm_checkpoint, self.tm_checkpoint)
            self.transition_model = pickle.loads(tm_checkpoint)
        else:
            self.transition_model.process(self._starting_state, learn=False)

    @property
    def name(self):
        return 'dreaming_double'
