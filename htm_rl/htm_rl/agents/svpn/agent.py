from typing import Dict, Tuple, Optional

import numpy as np
from numpy.random import Generator

from htm_rl.agents.agent import Wrapper
from htm_rl.agents.svpn.model import TransitionModel, RewardModel
from htm_rl.agents.svpn.sparse_value_network import SparseValueNetwork
from htm_rl.agents.ucb.agent import UcbAgent
from htm_rl.agents.ucb.sparse_value_network import exp_decay
from htm_rl.common.sdr import SparseSdr
from htm_rl.common.utils import isnone, clip
from htm_rl.envs.env import Env
from htm_rl.htm_plugins.temporal_memory import TemporalMemory


class Dreamer:
    svn: SparseValueNetwork

    planning_prob_alpha: Tuple[float, float]
    learning_rate: Tuple[float, float]
    td_lambda: bool
    trace_decay: float
    cell_eligibility_trace: Optional[np.ndarray]
    starting_sa_sdr: Optional[SparseSdr]
    TD_error: float

    def __init__(
            self, svn: SparseValueNetwork,
            planning_prob_alpha: Tuple[float, float],
            learning_rate: Tuple[float, float], trace_decay: Optional[float]
    ):
        if isinstance(learning_rate, float):
            self.learning_rate = svn.learning_rate[0] * learning_rate, svn.learning_rate[1]
        else:
            self.learning_rate = svn.learning_rate

        if trace_decay is None:
            trace_decay = svn.trace_decay

        self.planning_prob_alpha = planning_prob_alpha
        self.td_lambda = trace_decay > .0
        self.trace_decay = trace_decay

        self.svn = svn
        self.cell_eligibility_trace = None
        self.starting_sa_sdr = None
        self.TD_error = None

    def put_into_dream(self, starting_sa_sdr):
        wake_svn = self.svn
        wake_svn.learning_rate, self.learning_rate = self.learning_rate, wake_svn.learning_rate
        wake_svn.trace_decay, self.trace_decay = self.trace_decay, wake_svn.trace_decay

        self.cell_eligibility_trace = wake_svn.cell_eligibility_trace.copy()
        wake_svn.cell_eligibility_trace.fill(.0)

        self.starting_sa_sdr = starting_sa_sdr.copy()
        self.TD_error = wake_svn.TD_error

    def reset_dreaming(self):
        dreaming_svn = self.svn
        dreaming_svn.cell_eligibility_trace = self.cell_eligibility_trace.copy()
        dreaming_svn.cell_eligibility_trace.fill(.0)
        return self.starting_sa_sdr.copy()

    def wake(self):
        dreaming_svn = self.svn
        dreaming_svn.learning_rate, self.learning_rate = self.learning_rate, dreaming_svn.learning_rate
        dreaming_svn.trace_decay, self.trace_decay = self.trace_decay, dreaming_svn.trace_decay
        dreaming_svn.TD_error = self.TD_error
        dreaming_svn.cell_eligibility_trace = self.cell_eligibility_trace.copy()

    def decay_learning_factors(self):
        self.learning_rate = exp_decay(self.learning_rate)


class SvpnAgent(UcbAgent):
    """Sparse Value Prediction Network Agent"""
    SparseValueNetwork = SparseValueNetwork

    sqvn: SparseValueNetwork
    sa_transition_model: TransitionModel
    reward_model: RewardModel
    dreamer: Dreamer

    first_planning_episode: int
    last_planning_episode: Optional[int]
    prediction_depth: int
    n_prediction_rollouts: int
    dream_length: Optional[int]

    rng: Generator
    _episode: int

    def __init__(
            self, env: Env, seed: int,
            tm: Dict,
            dreamer: Dict,
            prediction_depth: int,
            n_prediction_rollouts: int,
            first_planning_episode: int,
            last_planning_episode: int = None,
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
        self.first_planning_episode = first_planning_episode
        self.last_planning_episode = isnone(last_planning_episode, 999999)
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
            self.reward_model.update(s, reward)
            # condition is to prevent inevitable useless planning in the end
            if reward <= 0:
                self._dream(s)

        if not first:
            # process feedback
            self._learn_step(
                prev_sa_sdr=self._current_sa_sdr,
                reward=reward,
                actions_sa_sdr=actions_sa_sdr
            )

        action = self.sqvn.choose(actions_sa_sdr)
        self._current_sa_sdr = actions_sa_sdr[action]
        self._process_transition(s, action, learn_tm=True)
        return action

    def _dream(self, starting_state):
        if not self._decide_to_dream():
            return

        # print(self.sqvn.TD_error)
        self.dreamer.planning_prob_alpha = exp_decay(self.dreamer.planning_prob_alpha)
        self._put_into_dream()

        for _ in range(self.n_prediction_rollouts):
            state = starting_state
            for i in range(self.prediction_depth):
                if len(state) == 0:
                    break
                state = self._move_in_dream(state, td_lambda=self.dreamer.td_lambda)
                self.dream_length += 1

            self._reset_dreaming()
            # print(i)

        self._wake()

    def _put_into_dream(self):
        self.sa_transition_model.save_tm_state()
        self.dreamer.put_into_dream(starting_sa_sdr=self._current_sa_sdr)
        self._reset_dreaming()
        self.dream_length = 0

    def _reset_dreaming(self):
        self._current_sa_sdr = self.dreamer.reset_dreaming()
        self.sa_transition_model.restore_tm_state()

    def _wake(self):
        self.dreamer.wake()

    def _move_in_dream(self, state: SparseSdr, td_lambda: bool):
        reward = self.reward_model.rewards[state].mean()
        actions_sa_sdr = self._encode_actions(state, learn=False)

        if self.rng.random() < .5:
            action = self.rng.choice(len(actions_sa_sdr))
        else:
            action = self.sqvn.choose(actions_sa_sdr)

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
        next_state = self._extract_state_from_s_a(s_a_next_superposition)
        return next_state

    def _choose_action(self, actions):
        return self.rng.choice(len(actions))

    def _decide_to_dream(self):
        self.dream_length = None
        if not self.first_planning_episode <= self.episode < self.last_planning_episode:
            return False
        if self.dreamer.learning_rate[0] < 1e-5:
            return False

        td_error = self.sqvn.TD_error
        planning_prob = clip(abs(td_error) / 2 - .05, 1.)
        # increase prob
        if planning_prob > .0:
            prob_boost = self.dreamer.planning_prob_alpha[0]
            planning_prob = 1 - (1 - planning_prob) / prob_boost
        return self.rng.random() < planning_prob

    def _process_transition(self, s, action, learn_tm: bool) -> Tuple[SparseSdr, SparseSdr]:
        a = self.action_encoder.encode(action)
        s_a = self.sa_concatenator.concatenate(s, a)

        return self.sa_transition_model.process(s_a, learn=learn_tm)

    def _extract_state_from_s_a(self, s_a: SparseSdr):
        size = self.state_sp.output_sdr_size
        state = np.array([x for x in s_a if x < size])
        return state

    def _on_new_episode(self):
        super(SvpnAgent, self)._on_new_episode()
        self.reward_model.decay_learning_factors()
        self.dreamer.decay_learning_factors()
        self.episode += 1

    @staticmethod
    def _make_sa_transition_model(state_encoder, action_encoder, tm_config):
        a_active_bits = action_encoder.output_sdr_size / action_encoder.n_values
        sa_active_bits = state_encoder.n_active_bits + a_active_bits
        tm = TemporalMemory(
            n_columns=action_encoder.output_sdr_size + state_encoder.output_sdr_size,
            n_active_bits=sa_active_bits,
            **tm_config
        )
        return TransitionModel(tm, collect_anomalies=True)


class ValueRecorder(Wrapper):
    root_agent: SvpnAgent

    def get_info(self) -> dict:
        res = self.agent.get_info()

        sa_sdr = self.root_agent._current_sa_sdr
        if sa_sdr is not None:
            res['value'] = self.root_agent.sqvn._value_option(sa_sdr, greedy=True)
        return res


class TDErrorRecorder(Wrapper):
    root_agent: SvpnAgent

    def get_info(self) -> dict:
        res = self.agent.get_info()
        res['td_error'] = self.root_agent.dreamer.TD_error
        return res


class AnomalyRecorder(Wrapper):
    root_agent: SvpnAgent

    def get_info(self) -> dict:
        res = self.agent.get_info()
        res['anomaly'] = self.root_agent.sa_transition_model.tm.anomaly
        return res


class DreamingRecorder(Wrapper):
    root_agent: SvpnAgent

    def get_info(self) -> dict:
        res = self.agent.get_info()
        res['dream_length'] = self.root_agent.dream_length
        return res
