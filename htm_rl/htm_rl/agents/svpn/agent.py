from typing import Dict, Tuple, Optional

import numpy as np
from numpy.random import Generator

from htm_rl.agents.svpn.model import TransitionModel
from htm_rl.agents.svpn.sparse_value_network import SparseValueNetwork
from htm_rl.agents.ucb.agent import UcbAgent
from htm_rl.common.sdr import SparseSdr
from htm_rl.common.utils import isnone, clip
from htm_rl.envs.env import Env
from htm_rl.htm_plugins.temporal_memory import TemporalMemory


class SvpnAgent(UcbAgent):
    """Sparse Value Prediction Network Agent"""
    SparseValueNetwork = SparseValueNetwork

    sa_transition_model: TransitionModel
    reward_model: np.ndarray

    r_learning_rate: float
    first_planning_episode: int
    last_planning_episode: Optional[int]
    prediction_depth: int
    n_prediction_rollouts: int

    rng: Generator
    _episode: int

    def __init__(
            self, env: Env, seed: int,
            tm: Dict,
            prediction_depth: int,
            n_prediction_rollouts: int,
            first_planning_episode: int,
            last_planning_episode: int = None,
            r_learning_rate: float = None,
            **ucb_agent_kwargs
    ):
        super().__init__(env, seed, **ucb_agent_kwargs)

        self.sa_transition_model = self._make_sa_transition_model(
            self.state_sp, self.action_encoder, tm
        )
        self.reward_model = np.zeros(self.state_sp.output_sdr_size, dtype=np.float)

        self.prediction_depth = prediction_depth
        self.n_prediction_rollouts = n_prediction_rollouts
        self.first_planning_episode = first_planning_episode
        self.last_planning_episode = isnone(last_planning_episode, 999999)
        self.r_learning_rate = r_learning_rate
        self.episode = 0
        self.cnt = 0
        self.rng = np.random.default_rng(seed)

    @property
    def name(self):
        return 'svpn'

    def act(self, reward: float, state: SparseSdr, first: bool):
        s = self.state_sp.compute(state, learn=True)

        # 2nd condition - to prevent inevitable useless planning in the end
        if not first or reward <= 0:
            self._update_reward_model(s, reward)
            if self.first_planning_episode <= self.episode < self.last_planning_episode:
                for i in range(self.n_prediction_rollouts):
                    self.dream(reward, s, depth=self.prediction_depth)
        else:
            self.episode += 1
            self.sqvn.reset()
            self.cnt = 0

        actions = self._encode_actions(s)
        action = self.sqvn.choose(actions)

        if not first:
            # process feedback
            prev_sa_sdr = self._current_sa
            greedy_action = self.sqvn.choose(actions, greedy=True)
            greedy_sa_sdr = actions[greedy_action]

            self.sqvn.update(
                sa=prev_sa_sdr,
                reward=reward,
                sa_next=greedy_sa_sdr,
                td_lambda=True
            )

        self._current_sa = actions[action]
        self._update_transition_model(s, action, learn_tm=True)
        return action

    def dream(self, reward: float, state: SparseSdr, depth: int):
        dreaming = False
        lr_factor = 2.
        init_len = len(state)
        for i in range(depth):
            if len(state) < .7 * init_len:
                break

            if i == 0:
                td_error = self.sqvn.TD_error
                planning_prob = clip(abs(td_error) / 2., 1.)
                if not self.rng.random() < planning_prob:
                    break

                self.cnt += 1
                self.sa_transition_model.save_tm_state()
                backup_sa = self._current_sa
                self.sqvn.learning_rate = (
                    self.sqvn.learning_rate[0] / lr_factor,
                    self.sqvn.learning_rate[1]
                )
                dreaming = True

            reward = self.reward_model[state].mean()
            action = self._act_in_dream(reward, state)

            _, s_a_next_superposition = self._update_transition_model(state, action, learn_tm=False)
            s_a_next_superposition = self.sa_transition_model.columns_from_cells(s_a_next_superposition)
            state = self._extract_state_from_s_a(s_a_next_superposition)

        if dreaming:
            self.sqvn.learning_rate = (
                self.sqvn.learning_rate[0] * lr_factor,
                self.sqvn.learning_rate[1]
            )
            self._current_sa = backup_sa
            self.sa_transition_model.restore_tm_state()
            # print(i)

    def _act_in_dream(self, reward, s: SparseSdr):
        actions = self._encode_actions(s)
        if self.rng.random() < .1:
            action = self.rng.choice(len(actions))
        else:
            action = self.sqvn.choose(actions)

        # process feedback
        prev_sa_sdr = self._current_sa
        greedy_action = self.sqvn.choose(actions, greedy=True)
        greedy_sa_sdr = actions[greedy_action]

        self.sqvn.update(
            sa=prev_sa_sdr,
            reward=reward,
            sa_next=greedy_sa_sdr,
            td_lambda=False
        )

        self._current_sa = actions[action]
        return action

    def _choose_action(self, actions):
        return self.rng.choice(len(actions))

    def _update_transition_model(self, s, action, learn_tm) -> Tuple[SparseSdr, SparseSdr]:
        a = self.action_encoder.encode(action)
        s_a = self.sa_concatenator.concatenate(s, a)

        return self.sa_transition_model.process(s_a, learn=learn_tm)

    def _update_reward_model(self, s: SparseSdr, reward: float):
        r_learning_rate = isnone(self.r_learning_rate, self.sqvn.learning_rate[0])
        self.reward_model *= 1 - r_learning_rate
        self.reward_model[s] += reward

    def _extract_state_from_s_a(self, s_a: SparseSdr):
        size = self.state_sp.output_sdr_size
        state = np.array([x for x in s_a if x < size])
        return state

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
