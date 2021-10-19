import pickle
from typing import Optional

import numpy as np
from numpy.random import Generator

from htm_rl.agents.dreamer.dreaming_stats import DreamingStats
from htm_rl.agents.dreamer.falling_asleep import AnomalyBasedFallingAsleep
from htm_rl.agents.dreamer.moving_average_tracker import MovingAverageTracker
from htm_rl.agents.q.cluster_memory import ClusterMemory
from htm_rl.agents.q.input_changes_detector import InputChangesDetector
from htm_rl.agents.qmb.anomaly_model import AnomalyModel
from htm_rl.agents.qmb.reward_model import RewardModel
from htm_rl.agents.qmb.transition_model import TransitionModel
from htm_rl.agents.qmb.transition_models import make_transition_model
from htm_rl.common.sdr import SparseSdr
from htm_rl.common.utils import isnone, safe_divide
from htm_rl.modules.dreaming.sa_encoder import DreamerSaEncoder


class Dreamer:
    """
    Dreamer object, that can substitute with itself the environment for an agent,
    such that the agent <--> env interaction is replaced with the
    agent <--> imaginary env (learned transition + reward models).
    """
    n_actions: int
    input_changes_detector: InputChangesDetector
    sa_encoder: DreamerSaEncoder

    # TM learns s_a -> s'
    transition_model: TransitionModel
    reward_model: RewardModel
    anomaly_model: AnomalyModel

    # turning it off disables possibility of dreaming
    enabled: bool
    # max prediction depth
    prediction_depth: int
    # min, max number of rollouts
    max_n_rollouts: int

    anomaly_based_falling_asleep: AnomalyBasedFallingAsleep

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
    total_stats: DreamingStats
    last_matched_mismatched: tuple[int, int]
    dreaming_match_rate_ma_tracker: MovingAverageTracker
    reward_ma_tracker: MovingAverageTracker
    anomaly_error_ma_tracker: MovingAverageTracker
    anticipation: float

    _prev_action: Optional[int]
    _starting_state: Optional[SparseSdr]
    _starting_s: Optional[SparseSdr]
    _tm_checkpoint: Optional[bytes]
    _episode: int
    _rng: Generator

    def __init__(
            self,
            # --- specify this
            seed: int,
            n_actions: int,  # env.n_actions
            agent,           # HIMA agent
            state_encoder,   # state --> s encoder
            # ----------------
            # --- from config
            sa_encoder: dict,
            transition_model: dict,
            reward_model: dict,
            anomaly_model: dict,
            anomaly_based_falling_asleep: dict,
            prediction_depth: int,
            max_n_rollouts: int,
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
    ):
        self.n_actions = n_actions
        self.agent = agent
        self.sa_encoder = DreamerSaEncoder(
            state_encoder, n_actions, **sa_encoder
        )
        self.input_changes_detector = InputChangesDetector(
            input_sdr_size=self.sa_encoder.state_sp.input_sdr_size
        )
        self.transition_model = make_transition_model(
            sa_encoder=self.sa_encoder,
            transition_model_config=transition_model
        )
        self.reward_model = RewardModel(
            cells_sdr_size=self.sa_encoder.s_output_sdr_size,
            **reward_model
        )
        self.anomaly_model = AnomalyModel(
            cells_sdr_size=self.sa_encoder.s_output_sdr_size,
            n_actions=self.n_actions,
            **anomaly_model
        )

        self.enabled = enabled
        self.anomaly_based_falling_asleep = AnomalyBasedFallingAsleep(
            **anomaly_based_falling_asleep
        )
        self.prediction_depth = prediction_depth
        self.max_n_rollouts = max_n_rollouts

        self.last_matched_mismatched = (0, 0)
        self.dreaming_match_rate_ma_tracker = MovingAverageTracker(*dreaming_match_rate_ma_tracker)
        self.reward_ma_tracker = MovingAverageTracker(*reward_ma_tracker)
        self.anomaly_error_ma_tracker = MovingAverageTracker(*anomaly_error_ma_tracker)
        self.anticipation = 0.
        self.dreaming_aug = 0.

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

        self._prev_action = None
        self._starting_state = None
        self._starting_s = None
        self._tm_checkpoint = None
        self._rng = np.random.default_rng(seed)
        self._episode = 0

        self.stats = DreamingStats(self._cluster_memory.stats)
        self.total_stats = DreamingStats()

    def on_wake_step(self, state: SparseSdr, reward: float, action: int):
        """
        Callback that should be called after each agent's step.
        It's used to learn transition, reward and anomaly models.

        :param state: raw state (observation) sparse sdr
        :param reward: reward value given for getting to the state `state`
        :param action: action index
        """
        # allows us to turn off learning after several input repeats
        input_changed = self.input_changes_detector.changed(state, train=True)

        s = self.sa_encoder.encode_state(state, learn=True and input_changed)
        first = self._prev_action is None

        if not first:
            # learn transition, anomaly and reward for (s',a') -> s
            # applicable only starting from the 2nd step
            self._on_transition_to_new_state(
                self._prev_action, s, reward, learn=True and input_changed
            )

        # activate (s,a) w/o learning
        self._on_action_selection(s, action)
        self._prev_action = action
        self.stats.add_step()
        self.total_stats.add_step()

    def on_new_episode(self):
        """
        Callback that should be called after the end of the episode.
        It's used to reset transition memory, increment episodes counter
        and decay some hyper-parameters.
        """
        self._episode += 1
        self._prev_action = None
        # no need to reset Transition Model - it's reset with
        # the first (s, a) [non-learnable] activation
        self.reward_model.decay_learning_factors()
        self.anomaly_model.decay_learning_factors()

    def on_new_goal(self):
        self.stats.reset()

    def can_dream(self, reward: float) -> bool:
        """
        Checks whether the dreaming is possible.
        """
        if not self.enabled:
            return False
        if reward > .2:
            # reward condition prevents useless planning when we've already
            # found the goal
            return False
        return True

    def decide_to_dream(self, state: SparseSdr, anomaly=None) -> bool:
        """
        Informs whether the dreaming should be activated.
        """
        if anomaly is None:
            s = self.sa_encoder.encode_state(state, learn=False)
            anomaly = self.anomaly_model.state_anomaly(s)
        return self._anomaly_based_dreaming(anomaly)

    def dream(self, starting_state: SparseSdr):
        """
        Switches to dreaming mode and performs several imaginary trajectory
        rollouts. Dreaming starts from `starting_state`, which is the current
        raw observation, for which the agent isn't acted yet.

        # TODO: you probably want to support callback that should be called after
        # TODO: each rollout to reset agent's state to the initial starting state
        """
        starting_s = self.sa_encoder.encode_state(starting_state, learn=False)
        self._put_into_dream(starting_state, starting_s)

        starting_anomaly = self.anomaly_model.state_anomaly(starting_s)
        i_rollout, goal_reached = 0, False
        sum_depth, sum_depth_smoothed, depths = 0, 0, []

        # loop over separate rollouts
        while i_rollout < self.max_n_rollouts:
            if i_rollout > 0 and not self.decide_to_dream(starting_state, anomaly=starting_anomaly):
                break
            self._on_new_rollout(i_rollout)
            state, s = starting_state, starting_s
            depth = 0
            # loop over one rollout's trajectory states
            while depth < self.prediction_depth:
                next_state, next_s, a, anomaly, goal_reached = self._move_in_dream(state, s)
                depth += 1

                if len(next_s) < .6 * len(self._starting_s) or anomaly > .7:
                    # predicted pattern is too vague -> lower than TM's threshold
                    # or anomaly is too high -> high risk for learning on garbage
                    break
                state, s = next_state, next_s

            i_rollout += 1
            sum_depth += depth
            sum_depth_smoothed += depth ** .6
            # depths.append(depth)
            self._on_complete_rollout(depth, goal_reached)

        # if depths: print(depths)
        self.stats.on_dreamed(i_rollout, sum_depth)
        self.total_stats.on_dreamed(i_rollout, sum_depth)
        self._wake()

    def _move_in_dream(self, state: SparseSdr, s: SparseSdr):
        reward = self.reward_model.state_reward(s)

        action = self.agent.make_action(state)
        self.agent.reinforce(reward)

        if reward > 2 * self.on_awake_step_surprise_threshold:
            # found goal ==> should stop rollout
            state_next, s_next, anomaly = [], [], 1.
            return state_next, s_next, action, anomaly, True

        # (s', a') -> s
        self.transition_model.process(s, learn=False)
        # s -> (s, a)
        s_a = self.sa_encoder.concat_s_action(s, action, learn=False)
        self.transition_model.process(s_a, learn=False)
        s_next = self.transition_model.predicted_cols

        if len(s_next) / len(self._starting_s) > 1.2:
            # if superposition grew too much, subsample it
            s_next = self._rng.choice(
                s_next, int(1.2 * len(self._starting_s)),
                replace=False
            )
            s_next.sort()

        # noinspection PyUnresolvedReferences
        if self.sa_encoder.state_clusters is not None:
            # noinspection PyUnresolvedReferences
            s_next = self.sa_encoder.restore_s(s_next)

        state_next = self.sa_encoder.decode_s_to_state(s_next)
        anomaly = self.anomaly_model.state_anomaly(s_next, prev_action=action)
        return state_next, s_next, action, anomaly, False

    def _put_into_dream(self, starting_state: SparseSdr, starting_s: SparseSdr):
        self._save_tm_checkpoint()
        self._starting_state = starting_state.copy()
        self._starting_s = starting_s.copy()
        self._cluster_memory.stats = self.stats.dreaming_cluster_memory_stats

    def _on_new_rollout(self, i_rollout):
        if i_rollout > 0:
            self._restore_tm_checkpoint()

    def _wake(self):
        self._cluster_memory.stats = self.stats.wake_cluster_memory_stats
        self._on_wake()
        self._restore_tm_checkpoint()

    def _anomaly_based_dreaming(self, anomaly):
        params = self.anomaly_based_falling_asleep

        if anomaly > params.anomaly_threshold.value:
            return False

        # p in [0, 1]
        p = 1 - anomaly

        # alpha in [0, 1] <=> break point p value
        alpha, beta = params.breaking_point, params.power
        # make non-linear by boosting p > alpha and inhibiting p < alpha, if b > 1
        # --> [0, > 1]
        p = (p / alpha)**beta
        # re-normalize back --> [0, 1]
        p /= (1 / alpha)**beta

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

    def _on_transition_to_new_state(
            self, prev_action: int, s: SparseSdr, reward: float, learn: bool
    ):
        # learn transition and anomaly for (s',a') -> s
        self.transition_model.process(s, learn=learn)
        if not learn:
            return

        self.anomaly_model.update(prev_action, s, self.transition_model.anomaly)
        # also update reward model
        self.reward_model.update(s, reward)

        # ==========================================

        anomaly_error = self.anomaly_model.last_error
        self.reward_ma_tracker.update(reward)
        self.anomaly_error_ma_tracker.update(anomaly_error)

        # return
        dreaming_probability = self.anomaly_based_falling_asleep.probability
        anomaly_threshold = self.anomaly_based_falling_asleep.anomaly_threshold

        anomaly_error_short_ma = self.anomaly_error_ma_tracker.short_ma
        anomaly_error_long_ma = self.anomaly_error_ma_tracker.long_ma
        certainty = 1 - abs(anomaly_error_short_ma)

        if abs(self.reward_model.last_error) * certainty > self.on_awake_step_surprise_threshold:
            # surprising event
            self.anticipation = 1.
            dreaming_probability.update(dreaming_probability.baseline)
        elif reward > self.on_awake_step_surprise_threshold:
            # expected reward
            self.anticipation *= self.on_awake_step_anticipation_decay
            dreaming_probability.update(
                dreaming_probability.min_value +
                (dreaming_probability.max_value - dreaming_probability.min_value) * self.anticipation
            )
        else:
            relax_to_baseline_scale = 1 - (1 - self.on_awake_step_anticipation_decay) / 15
            delta_baseline = dreaming_probability.value - dreaming_probability.baseline
            dreaming_probability.update(
                dreaming_probability.baseline +
                delta_baseline * relax_to_baseline_scale
            )

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

    # def _on_transition_to_new_state(
    #         self, prev_action: int, s: SparseSdr, reward: float, learn: bool
    # ):
    #     # learn transition and anomaly for (s',a') -> s
    #     self.transition_model.process(s, learn=learn)
    #     if not learn:
    #         return
    #
    #     self.anomaly_model.update(prev_action, s, self.transition_model.anomaly)
    #     # also update reward model
    #     self.reward_model.update(s, reward)
    #
    #     # ==========================================
    #
    #     anomaly_error = self.anomaly_model.last_error
    #     self.reward_ma_tracker.update(reward)
    #     self.anomaly_error_ma_tracker.update(anomaly_error)
    #
    #     # return
    #     dreaming_probability = self.anomaly_based_falling_asleep.probability
    #     anomaly_threshold = self.anomaly_based_falling_asleep.anomaly_threshold
    #
    #     anomaly_error_short_ma = self.anomaly_error_ma_tracker.short_ma
    #     anomaly_error_long_ma = self.anomaly_error_ma_tracker.long_ma
    #     certainty = 1 - abs(anomaly_error_short_ma)
    #     longtime_certainty = 1 - abs(anomaly_error_long_ma)
    #
    #     if abs(self.reward_model.last_error) * certainty > self.on_awake_step_surprise_threshold:
    #         # surprising event
    #         self.anticipation = 1.
    #         self.dreaming_aug = self.anticipation
    #     elif reward > self.on_awake_step_surprise_threshold:
    #         self.anticipation *= self.on_awake_step_anticipation_decay
    #
    #     reward_short_ma = self.reward_ma_tracker.short_ma
    #     reward_long_ma = self.reward_ma_tracker.long_ma
    #     r_eps = self.on_awake_step_reward_ma_eps
    #     anticipation_scale = self.on_awake_step_prob_delta_anticipation_scale
    #     # alpha = self.dreaming_aug * 1.4 * (1 - self.on_awake_step_anticipation_decay)
    #     alpha = self.anticipation
    #     if reward_short_ma > reward_long_ma + r_eps:
    #         # good place
    #         dreaming_probability.add(
    #             anticipation_scale * alpha * longtime_certainty
    #         )
    #     elif reward_short_ma < reward_long_ma - r_eps:
    #         # lackluster
    #         dreaming_probability.add(
    #             -.9 * anticipation_scale * alpha * longtime_certainty
    #         )
    #
    #     an_eps = self.on_awake_step_anomaly_err_ma_eps
    #     if abs(anomaly_error_short_ma) + an_eps < abs(anomaly_error_long_ma):
    #         # short-radius lower anomaly err
    #         anomaly_threshold.add(self.on_awake_step_ma_based_an_th_delta)
    #     elif abs(anomaly_error_short_ma) > abs(anomaly_error_long_ma) + an_eps:
    #         # short-radius higher anomaly err
    #         anomaly_threshold.add(-self.on_awake_step_ma_based_an_th_delta)
    #     elif abs(anomaly_error_long_ma) < self.on_awake_step_long_ma_based_an_th_threshold:
    #         # long-radius good anomaly approx - stable encoding
    #         anomaly_threshold.add(self.on_awake_step_long_ma_based_an_th_delta)
    #     else:
    #         # long-radius bad anomaly approx
    #         anomaly_threshold.add(-self.on_awake_step_long_ma_based_an_th_delta)

    def _on_wake(self):
        dcl_stats = self.stats.dreaming_cluster_memory_stats

        last_matched, last_mismatched = self.last_matched_mismatched
        matched, mismatched = dcl_stats.matched, dcl_stats.mismatched
        instant_match_rate = safe_divide(
            matched - last_matched,
            matched - last_matched + mismatched - last_mismatched
        )
        self.dreaming_match_rate_ma_tracker.update(instant_match_rate)

    def _on_complete_rollout(self, depth, goal_reached):
        ...

    def _on_action_selection(self, s: SparseSdr, action: int):
        # activate (s,a) w/o learning
        s_a = self.sa_encoder.concat_s_action(s, action, learn=False)
        self.transition_model.process(s_a, learn=False)

    @property
    def _cluster_memory(self) -> ClusterMemory:
        return self.sa_encoder.state_clusters

    @property
    def name(self):
        return 'dreamer'

    def log_stats(self, wandb_run, step):
        if not self.enabled:
            return

        stats, total_stats = self.stats, self.total_stats
        stats_to_log = {
            'rollouts': stats.rollouts,
            'avg_dreaming_rate': safe_divide(stats.dreaming_times, stats.wake_steps),
            'avg_depth': stats.avg_depth,
            'dreaming_probability': self.anomaly_based_falling_asleep.probability.value,
            'anomaly_threshold': self.anomaly_based_falling_asleep.anomaly_threshold.value,
            'total_rollouts': total_stats.rollouts,
            'total_avg_dreaming_rate': safe_divide(total_stats.dreaming_times, total_stats.wake_steps),
            'total_avg_depth': total_stats.avg_depth,
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
