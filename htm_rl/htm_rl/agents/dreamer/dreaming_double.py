import pickle
from typing import Optional

import numpy as np
from numpy.random import Generator

from htm_rl.agents.dreamer.dreaming_stats import DreamingStats
from htm_rl.agents.dreamer.falling_asleep import TdErrorBasedFallingAsleep, AnomalyBasedFallingAsleep
from htm_rl.agents.dreamer.moving_average_tracker import MovingAverageTracker
from htm_rl.agents.dreamer.qvn_double import QValueNetworkDouble
from htm_rl.agents.q.cluster_memory import ClusterMemory
from htm_rl.agents.q.eligibility_traces import EligibilityTraces
from htm_rl.agents.qmb.agent import QModelBasedAgent
from htm_rl.common.sdr import SparseSdr
from htm_rl.common.utils import multiply_decaying_value, exp_decay, isnone, DecayingValue, safe_divide


class DreamingDouble(QModelBasedAgent):
    Q: QValueNetworkDouble
    wake_E_traces: EligibilityTraces
    derive_E_traces: bool
    rollout_q_lr_decay_power: float
    trajectory_exploration_eps_decay: float

    enabled: bool
    falling_asleep_strategy: str
    td_error_based_falling_asleep: Optional[TdErrorBasedFallingAsleep]
    anomaly_based_falling_asleep: Optional[AnomalyBasedFallingAsleep]
    prediction_depth: int
    n_prediction_rollouts: tuple[int, int]

    on_wake_dreaming_match_rate_threshold: float
    on_wake_dreaming_match_rate_prob_inh: float
    on_wake_dreaming_match_rate_an_th_inh: float
    on_wake_prob_decay: float

    on_awake_step_surprise_threshold: float
    on_awake_step_anticipation_decay: float
    on_awake_step_reward_ma_eps: float
    on_awake_step_prob_delta_anticipation_scale: float
    on_awake_step_anomaly_err_ma_eps: float
    on_awake_step_ma_based_an_th_delta: float
    on_awake_step_long_ma_based_an_th_threshold: float
    on_awake_step_long_ma_based_an_th_delta: float

    stats: DreamingStats
    last_matched_mismatched: tuple[int, int]
    dreaming_match_rate_ma_tracker: MovingAverageTracker
    reward_ma_tracker: MovingAverageTracker
    anomaly_error_ma_tracker: MovingAverageTracker
    anticipation: float

    _starting_sa_sdr: Optional[SparseSdr]
    _starting_state: Optional[SparseSdr]
    _tm_checkpoint: Optional[bytes]
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
            dreaming_match_rate_ma_tracker: list[int],
            reward_ma_tracker: list[int],
            anomaly_error_ma_tracker: list[int],
            on_wake_dreaming_match_rate_threshold: float,
            on_wake_dreaming_match_rate_prob_inh: float,
            on_wake_dreaming_match_rate_an_th_inh: float,
            on_wake_prob_decay: float,

            on_awake_step_surprise_threshold: float,
            on_awake_step_anticipation_decay: float,
            on_awake_step_reward_ma_eps: float,
            on_awake_step_prob_delta_anticipation_scale: float,
            on_awake_step_anomaly_err_ma_eps: float,
            on_awake_step_ma_based_an_th_delta: float,
            on_awake_step_long_ma_based_an_th_threshold: float,
            on_awake_step_long_ma_based_an_th_delta: float,
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

        self.last_matched_mismatched = (0, 0)
        self.dreaming_match_rate_ma_tracker = MovingAverageTracker(*dreaming_match_rate_ma_tracker)
        self.reward_ma_tracker = MovingAverageTracker(*reward_ma_tracker)
        self.anomaly_error_ma_tracker = MovingAverageTracker(*anomaly_error_ma_tracker)
        self.anticipation = 0.

        self.on_wake_dreaming_match_rate_threshold = on_wake_dreaming_match_rate_threshold
        self.on_wake_dreaming_match_rate_prob_inh = on_wake_dreaming_match_rate_prob_inh
        self.on_wake_dreaming_match_rate_an_th_inh = on_wake_dreaming_match_rate_an_th_inh
        self.on_wake_prob_decay = on_wake_prob_decay

        self.on_awake_step_surprise_threshold = on_awake_step_surprise_threshold
        self.on_awake_step_anticipation_decay = on_awake_step_anticipation_decay
        self.on_awake_step_reward_ma_eps = on_awake_step_reward_ma_eps
        self.on_awake_step_prob_delta_anticipation_scale = on_awake_step_prob_delta_anticipation_scale
        self.on_awake_step_anomaly_err_ma_eps = on_awake_step_anomaly_err_ma_eps
        self.on_awake_step_ma_based_an_th_delta = on_awake_step_ma_based_an_th_delta
        self.on_awake_step_long_ma_based_an_th_threshold = on_awake_step_long_ma_based_an_th_threshold
        self.on_awake_step_long_ma_based_an_th_delta = on_awake_step_long_ma_based_an_th_delta

        self.train = True
        self._starting_sa_sdr = None
        self._starting_state = None
        self._tm_checkpoint = None
        self._exploration_eps_backup = None
        self._rng = np.random.default_rng(seed)
        self._current_sa_sdr = None
        self._step = 0
        self._episode = 0

        if self._cluster_memory is not None:
            self.stats = DreamingStats(self._cluster_memory.stats)
        else:
            self.stats = DreamingStats()

    def on_new_step(self):
        self.stats.add_step()

    def on_new_episode(self):
        self._episode += 1
        self.td_error_based_falling_asleep.boost_prob_alpha = exp_decay(
            self.td_error_based_falling_asleep.boost_prob_alpha
        )
        self.E_traces.decay_trace_decay()
        self.exploration_eps = exp_decay(self.exploration_eps)
        self.softmax_temp = exp_decay(self.softmax_temp)

        # self._print_dreaming_stats()

    def on_new_goal(self):
        self.stats.reset()

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

        i_rollout = 0
        min_n_rollouts, max_n_rollouts = self.n_prediction_rollouts
        sum_depth, sum_depth_smoothed, depths = 0, 0, []

        while i_rollout < min_n_rollouts or (
                i_rollout < max_n_rollouts and safe_divide(sum_depth_smoothed, i_rollout) >= 2.5
        ):
            self._on_new_rollout(i_rollout)
            state = starting_state
            depth = 0
            while depth < self.prediction_depth:
                next_state, a, anomaly = self._move_in_dream(state)
                depth += 1
                if len(next_state) < .6 * len(starting_state) or anomaly > .7:
                    break
                state = next_state

            i_rollout += 1
            sum_depth += depth
            sum_depth_smoothed += depth ** .6
            # depths.append(depth)
            self._on_complete_rollout(depth)

        # if depths: print(depths)
        self.stats.on_dreamed(i_rollout, sum_depth)
        self._wake()

    def _move_in_dream(self, state: SparseSdr):
        s = state
        reward = self.reward_model.state_reward(s)
        action = self.act(reward, s, first=False)

        if reward > .3:
            # found goal ==> should stop rollout
            return [], action, 1.

        # (s', a') -> s
        self.transition_model.process(s, learn=False)
        # s -> (s, a)
        s_a = self.sa_encoder.concat_s_action(s, action, learn=False)
        self.transition_model.process(s_a, learn=False)
        s_next = self.transition_model.predicted_cols

        if len(s_next) / len(self._starting_state) > 1.2:
            # if superposition grew too much, subsample it
            s_next = self._rng.choice(
                s_next, int(1.2 * len(self._starting_state)),
                replace=False
            )
            s_next.sort()

        # noinspection PyUnresolvedReferences
        if self.sa_encoder.state_clusters is not None:
            # noinspection PyUnresolvedReferences
            s_next = self.sa_encoder.restore_s(s_next)

        anomaly = self.anomaly_model.state_anomaly(s_next, prev_action=action)
        return s_next, action, anomaly

    def _put_into_dream(self, starting_state: SparseSdr, starting_sa_sdr: SparseSdr):
        self._save_tm_checkpoint()
        self._starting_state = starting_state.copy()
        self._starting_sa_sdr = starting_sa_sdr.copy()
        self._exploration_eps_backup = self.exploration_eps
        if self._cluster_memory is not None:
            self._cluster_memory.stats = self.stats.dreaming_cluster_memory_stats

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
        if self._cluster_memory is not None:
            self._cluster_memory.stats = self.stats.wake_cluster_memory_stats

        self._on_wake()
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

    def on_transition_to_new_state(self, reward: float):
        anomaly_error = self.anomaly_model.last_error
        self.reward_ma_tracker.update(reward)
        self.anomaly_error_ma_tracker.update(anomaly_error)

        # return
        dreaming_probability = self.anomaly_based_falling_asleep.probability
        anomaly_threshold = self.anomaly_based_falling_asleep.anomaly_threshold

        anomaly_error_short_ma = self.anomaly_error_ma_tracker.short_ma
        anomaly_error_long_ma = self.anomaly_error_ma_tracker.long_ma
        certainty = 1 - abs(anomaly_error_short_ma)
        longtime_certainty = 1 - abs(anomaly_error_long_ma)

        if abs(self.reward_model.last_error) * certainty > self.on_awake_step_surprise_threshold:
            # surprising event
            # print('+', certainty, self.reward_model.last_error)
            self.anticipation = 1.
            # if self.reward_model.last_error > on_awake_step_surprise_threshold:
            #     print('+', certainty, self.reward_model.last_error)
            #     on_awake_step_good_surprise_delta = .025
            #     dreaming_probability.add(on_awake_step_good_surprise_delta * certainty)
            # elif self.reward_model.last_error < -on_awake_step_surprise_threshold:
            #     print('-', certainty, self.reward_model.last_error)
            #     on_awake_step_bad_surprise_delta = .02
            #     dreaming_probability.add(on_awake_step_bad_surprise_delta * certainty)
        elif reward > self.on_awake_step_surprise_threshold:
            self.anticipation *= self.on_awake_step_anticipation_decay

        reward_short_ma = self.reward_ma_tracker.short_ma
        reward_long_ma = self.reward_ma_tracker.long_ma
        r_eps = self.on_awake_step_reward_ma_eps
        anticipation_scale = self.on_awake_step_prob_delta_anticipation_scale
        if reward_short_ma > reward_long_ma + r_eps:
            # good place
            dreaming_probability.add(
                anticipation_scale * self.anticipation * longtime_certainty
            )
        elif reward_short_ma < reward_long_ma - r_eps:
            # lackluster
            dreaming_probability.add(
                -anticipation_scale * self.anticipation * longtime_certainty
            )

        # if self.anomaly_error_ma_tracker.short_ma < self.anomaly_error_ma_tracker.long_ma - .02:
        #     anomaly_threshold.add(-.001)
        # elif self.anomaly_error_ma_tracker.short_ma > self.anomaly_error_ma_tracker.long_ma + .02:
        #     anomaly_threshold.add(.001)
        an_eps = self.on_awake_step_anomaly_err_ma_eps
        if abs(anomaly_error_short_ma) + an_eps < abs(anomaly_error_long_ma):
            # short-radius lower anomaly err
            anomaly_threshold.add(self.on_awake_step_ma_based_an_th_delta)
        elif abs(anomaly_error_short_ma) > abs(anomaly_error_long_ma) + an_eps:
            # short-radius higher anomaly err
            anomaly_threshold.add(-self.on_awake_step_ma_based_an_th_delta)
        elif abs(anomaly_error_long_ma) < self.on_awake_step_long_ma_based_an_th_threshold:
            # long-radius good anomaly approx - stable encoding
            anomaly_threshold.add(self.on_awake_step_long_ma_based_an_th_delta)
        else:
            # long-radius bad anomaly approx
            anomaly_threshold.add(-self.on_awake_step_long_ma_based_an_th_delta)

    def _on_wake(self):
        dcl_stats = self.stats.dreaming_cluster_memory_stats

        last_matched, last_mismatched = self.last_matched_mismatched
        matched, mismatched = dcl_stats.matched, dcl_stats.mismatched
        instant_match_rate = safe_divide(
            matched - last_matched,
            matched - last_matched + mismatched - last_mismatched
        )
        self.dreaming_match_rate_ma_tracker.update(instant_match_rate)

        dreaming_probability = self.anomaly_based_falling_asleep.probability
        anomaly_threshold = self.anomaly_based_falling_asleep.anomaly_threshold

        if self.dreaming_match_rate_ma_tracker.long_ma < self.on_wake_dreaming_match_rate_threshold:
            # bad TM ==>
            dreaming_probability.add(-self.on_wake_dreaming_match_rate_prob_inh)
            anomaly_threshold.add(-self.on_wake_dreaming_match_rate_an_th_inh)

        # relaxing dreaming probability decay
        dreaming_probability.scale(self.on_wake_prob_decay)

    def _on_complete_rollout(self, depth):
        ...

    def _td_error_based_dreaming(self, td_error):
        boost_prob_alpha = self.td_error_based_falling_asleep.boost_prob_alpha[0]
        prob_threshold = self.td_error_based_falling_asleep.prob_threshold
        max_abs_td_error = 2.
        p = (boost_prob_alpha * abs(td_error) - prob_threshold)
        p = np.clip(p / max_abs_td_error, 0., 1.)
        return self._rng.random() < p

    def _anomaly_based_dreaming(self, anomaly):
        params = self.anomaly_based_falling_asleep

        if anomaly > params.anomaly_threshold.value:
            return False

        # p in [0, 1]
        p = 1 - anomaly

        # alpha in [0, 1] <=> break point p value
        breaking_point, alpha = params.breaking_point, params.power
        # make non-linear by boosting p > alpha and inhibiting p < alpha, if b > 1
        # --> [0, > 1]
        p = (p / breaking_point)**alpha
        # re-normalize back --> [0, 1]
        p /= (1 / breaking_point)**alpha

        # shrink to max allowed probability --> [0, max_prob]
        p *= params.probability.value
        # sample
        return self._rng.random() < p

    def _save_tm_checkpoint(self) -> bytes:
        """Saves TM state."""
        if self.transition_model.tm.cells_per_column > 1:
            self._tm_checkpoint = pickle.dumps(self.transition_model)
            return self._tm_checkpoint

    def _restore_tm_checkpoint(self, tm_checkpoint: bytes = None):
        """Restores saved TM state."""
        if self.transition_model.tm.cells_per_column > 1:
            tm_checkpoint = isnone(tm_checkpoint, self._tm_checkpoint)
            self.transition_model = pickle.loads(tm_checkpoint)
        else:
            self.transition_model.process(self._starting_state, learn=False)

    @property
    def _cluster_memory(self) -> ClusterMemory:
        # noinspection PyUnresolvedReferences
        return self.sa_encoder.state_clusters

    @property
    def name(self):
        return 'dreaming_double'

    def _print_dreaming_stats(self):
        if self._episode == 0:
            return
        prob = self.anomaly_based_falling_asleep.probability
        threshold = self.anomaly_based_falling_asleep.anomaly_threshold

        print(
            'prob', round(100 * prob.value, 1), 'th', round(100 * threshold.value, 1),
            '| roll', self.stats.rollouts, 'dep', round(self.stats.avg_depth, 1),
            '| d a',
            round(isnone(self.dreaming_match_rate_ma_tracker.short_ma, 0), 2),
            round(isnone(self.dreaming_match_rate_ma_tracker.long_ma, 0), 2),
            # round(self.reward_ma_tracker.short_ma, 3), round(self.reward_ma_tracker.long_ma, 3),
            round(self.anomaly_error_ma_tracker.short_ma, 3),
            round(self.anomaly_error_ma_tracker.long_ma, 3),
        )
        # ds = self.stats.dreaming_cluster_memory_stats
        # ws = self.stats.wake_cluster_memory_stats
        # print(
        #     'wake rate sim',
        #     ws.matched + ws.mismatched, ws.matched,
        #     round(ws.avg_match_similarity, 3), round(ws.avg_mismatch_similarity, 3),
        #     ' | ',
        #     ws.added, ws.removed,
        #     ' | ',
        #     round(ws.avg_removed_cluster_intra_similarity, 3),
        #     round(ws.avg_removed_cluster_trace, 6)
        # )
        # print(
        #     'dreaming rate sim',
        #     ds.matched + ds.mismatched, ds.matched,
        #     round(ds.avg_match_similarity, 3), round(ds.avg_mismatch_similarity, 3),
        # )
        print('-----------------------')

    def log_stats(self, wandb_run, step, with_reset):
        if not self.enabled:
            return

        stats = self.stats
        stats_to_log = {
            'rollouts': stats.rollouts,
            'avg_dreaming_rate': safe_divide(stats.times, stats.steps),
            'avg_depth': stats.avg_depth,
            'dreaming_probability': self.anomaly_based_falling_asleep.probability.value,
            'anomaly_threshold': self.anomaly_based_falling_asleep.anomaly_threshold.value,
        }

        if stats.wake_cluster_memory_stats is not None:
            cl_wake_stats = stats.wake_cluster_memory_stats
            cl_dreaming_stats = stats.dreaming_cluster_memory_stats
            stats_to_log.update({
                'cl_wake_match_rate': cl_wake_stats.avg_match_rate,
                'cl_wake_avg_match_similarity': cl_wake_stats.avg_match_similarity,
                'cl_wake_avg_mismatch_similarity': cl_wake_stats.avg_mismatch_similarity,
                'cl_wake_all': cl_wake_stats.matched + cl_wake_stats.mismatched,
                'cl_wake_added': cl_wake_stats.added,
                'cl_wake_removed': cl_wake_stats.removed,
                'cl_wake_avg_removed_cluster_intra_similarity': cl_wake_stats.avg_removed_cluster_intra_similarity,
                'cl_wake_avg_removed_cluster_trace': cl_wake_stats.avg_removed_cluster_trace,
                'cl_dreaming_all': cl_dreaming_stats.matched + cl_dreaming_stats.mismatched,
                'cl_dreaming_match_rate': cl_dreaming_stats.avg_match_rate,
                'cl_dreaming_avg_match_similarity': cl_dreaming_stats.avg_match_similarity,
                'cl_dreaming_avg_mismatch_similarity': cl_dreaming_stats.avg_mismatch_similarity,
            })

        for k in stats_to_log.keys():
            wandb_run.log({f'dreaming/{k}': stats_to_log[k]}, step=step)

        if with_reset:
            self.on_new_goal()
