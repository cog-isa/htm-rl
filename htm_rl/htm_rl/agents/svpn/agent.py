from typing import Dict, Tuple, Optional

import numpy as np
from numpy.random import Generator

from htm_rl.agents.agent import Wrapper
from htm_rl.agents.svpn.model import TransitionModel, RewardModel
from htm_rl.agents.svpn.sparse_value_network import SparseValueNetwork
from htm_rl.agents.ucb.agent import UcbAgent
from htm_rl.agents.ucb.sparse_value_network import exp_decay, modify_factor_tuple
from htm_rl.common.sdr import SparseSdr
from htm_rl.common.utils import isnone, clip
from htm_rl.envs.env import Env
from htm_rl.htm_plugins.temporal_memory import TemporalMemory


class Dreamer:
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


class SvpnAgent(UcbAgent):
    """Sparse Value Prediction Network Agent"""
    SparseValueNetwork = SparseValueNetwork

    sqvn: SparseValueNetwork
    sa_transition_model: TransitionModel
    reward_model: RewardModel
    dreamer: Dreamer

    first_dreaming_episode: int
    last_dreaming_episode: Optional[int]
    prediction_depth: int
    n_prediction_rollouts: Tuple[int, int]
    dream_length: Optional[int]

    rng: Generator
    _episode: int

    def __init__(
            self, env: Env, seed: int,
            tm: Dict,
            dreamer: Dict,
            prediction_depth: int,
            n_prediction_rollouts: Tuple[int, int],
            first_dreaming_episode: int = 0,
            last_dreaming_episode: int = None,
            r_learning_rate: Tuple[float, float] = None,
            **ucb_agent_kwargs
    ):
        super().__init__(env, seed, **ucb_agent_kwargs)

        self.sa_transition_model = self._make_sa_transition_model(
            self.state_sp, self.action_encoder, tm
        )
        self.reward_model = RewardModel(
            self.state_sp.output_sdr_size,
            isnone(r_learning_rate, self.sqvn.learning_rate)
        )
        self.dreamer = Dreamer(self.sqvn, **dreamer)

        self.prediction_depth = prediction_depth
        self.n_prediction_rollouts = n_prediction_rollouts
        self.first_dreaming_episode = first_dreaming_episode
        self.last_dreaming_episode = isnone(last_dreaming_episode, 999999)
        self.dream_length = None
        self.episode = 0
        self.rng = np.random.default_rng(seed)

    @property
    def name(self):
        return 'svpn'

    def act(self, reward: float, state: SparseSdr, first: bool):
        if first:
            self._on_new_episode()

        s = self.state_sp.compute(state, learn=True)
        actions_sa_sdr = self._encode_actions(s, learn=True)

        if not first:
            # process feedback
            self._learn_step(
                prev_sa_sdr=self._current_sa_sdr,
                reward=reward,
                actions_sa_sdr=actions_sa_sdr
            )

        if not first:
            self.reward_model.update(s, reward)
            # condition prevents inevitable useless planning in the end
            if reward <= 0.:
                self._dream(s)

        action = self.sqvn.choose(actions_sa_sdr)
        self._current_sa_sdr = actions_sa_sdr[action]
        self._process_transition(s, action, learn_tm=True)
        return action

    def _dream(self, starting_state):
        if not self._decide_to_dream():
            return

        # print(self.sqvn.TD_error)
        self.dreamer.dreaming_prob_alpha = exp_decay(self.dreamer.dreaming_prob_alpha)
        self._put_into_dream()

        starting_state_len = len(starting_state)
        i_rollout = 0
        sum_depth = 0
        depths = []
        while i_rollout < self.n_prediction_rollouts[0] or (
                i_rollout < self.n_prediction_rollouts[1]
                and sum_depth * self.dreamer.rnd_move_prob >= .3 * i_rollout
        ):
            self._reset_dreaming(i_rollout)
            state = starting_state
            depth = 0
            for depth in range(self.prediction_depth):
                state = self._move_in_dream(state, td_lambda=self.dreamer.td_lambda)
                if len(state) < .6 * starting_state_len:
                    break

            sum_depth += depth ** .5
            i_rollout += 1
            depths.append(depth)

        self.dream_length += sum_depth
        if depths:
            print(depths)
        self._wake()

    def _put_into_dream(self):
        self.sa_transition_model.save_tm_state()
        self.dreamer.put_into_dream(starting_sa_sdr=self._current_sa_sdr)
        self.dream_length = 0

    def _reset_dreaming(self, i_rollout=None):
        if i_rollout == 0:
            return
        self._current_sa_sdr = self.dreamer.reset_dreaming(i_rollout)
        self.sa_transition_model.restore_tm_state()

    def _wake(self):
        self._reset_dreaming()
        self.dreamer.wake()

    def _move_in_dream(self, state: SparseSdr, td_lambda: bool):
        reward = self.reward_model.rewards[state].mean()
        actions_sa_sdr = self._encode_actions(state, learn=False)

        if self.rng.random() < self.dreamer.rnd_move_prob:
            action = self.rng.choice(self.n_actions)
        else:
            action = self.sqvn.choose(actions_sa_sdr, greedy=True)

        # process feedback
        self._learn_step(
            prev_sa_sdr=self._current_sa_sdr,
            reward=reward,
            actions_sa_sdr=actions_sa_sdr,
            td_lambda=td_lambda,
            update_visit_count=False
        )

        self._current_sa_sdr = actions_sa_sdr[action]
        _, s_a_next_superposition = self._process_transition(state, action, learn_tm=False)
        s_a_next_superposition = self.sa_transition_model.columns_from_cells(s_a_next_superposition)
        next_state = self._split_s_a(s_a_next_superposition)
        return next_state

    def _choose_action(self, actions):
        return self.rng.choice(len(actions))

    def _decide_to_dream(self):
        self.dream_length = None
        if not self.dreamer.enabled:
            return False
        if not self.first_dreaming_episode <= self.episode < self.last_dreaming_episode:
            return False
        if self.dreamer.learning_rate_factor < 1e-3:
            return False

        td_error = self.sqvn.TD_error
        max_abs_td_error = 2.
        dreaming_prob_boost = self.dreamer.dreaming_prob_alpha[0]
        dreaming_prob = (dreaming_prob_boost * abs(td_error) - self.dreamer.dreaming_prob_threshold)
        dreaming_prob = clip(dreaming_prob / max_abs_td_error, 1.)
        return self.rng.random() < dreaming_prob

    def _process_transition(self, s, action, learn_tm: bool) -> Tuple[SparseSdr, SparseSdr]:
        a = self.action_encoder.encode(action)
        s_a = self.sa_concatenator.concatenate(s, a)

        return self.sa_transition_model.process(s_a, learn=learn_tm)

    def _split_s_a(self, s_a: SparseSdr, with_options=False):
        size = self.state_sp.output_sdr_size
        state = np.array([x for x in s_a if x < size])
        if not with_options:
            return state

        actions = [x - size for x in s_a if x >= size]
        actions_activation = [0 for _ in range(self.n_actions)]
        for x in actions:
            actions_activation[self.action_encoder.decode_bit(x)] += 1

        actions = []
        for action, activation in enumerate(actions_activation):
            if self.action_encoder.activation_fraction(activation) > .3:
                actions.append(action)

        return state, actions

    def _on_new_episode(self):
        super(SvpnAgent, self)._on_new_episode()
        self.reward_model.decay_learning_factors()
        self.sa_transition_model.reset()
        self.episode += 1

    @staticmethod
    def _make_sa_transition_model(state_encoder, action_encoder, tm_config):
        a_active_bits = action_encoder.output_sdr_size / action_encoder.n_values
        sa_active_bits = state_encoder.n_active_bits + a_active_bits

        # print(state_encoder.output_sdr_size, state_encoder.n_active_bits, action_encoder.output_sdr_size, a_active_bits)
        tm = TemporalMemory(
            n_columns=action_encoder.output_sdr_size + state_encoder.output_sdr_size,
            n_active_bits=sa_active_bits,
            **tm_config
        )
        return TransitionModel(tm, collect_anomalies=True)


class ValueProvider(Wrapper):
    root_agent: SvpnAgent

    def get_info(self) -> dict:
        res = self.agent.get_info()

        sa_sdr = self.root_agent._current_sa_sdr
        if sa_sdr is not None:
            res['value'] = self.root_agent.sqvn._value_option(sa_sdr, greedy=True)
            # res['value_exp'] = self.root_agent.sqvn._value_option(sa_sdr, greedy=False)
        return res


class TDErrorProvider(Wrapper):
    root_agent: SvpnAgent

    def get_info(self) -> dict:
        res = self.agent.get_info()
        res['td_error'] = self.root_agent.dreamer.TD_error
        return res


class AnomalyProvider(Wrapper):
    root_agent: SvpnAgent

    def get_info(self) -> dict:
        res = self.agent.get_info()
        res['anomaly'] = self.root_agent.sa_transition_model.anomaly
        return res


class DreamingLengthProvider(Wrapper):
    root_agent: SvpnAgent

    def get_info(self) -> dict:
        res = self.agent.get_info()
        res['dream_length'] = self.root_agent.dream_length
        return res
