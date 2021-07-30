from typing import Optional, List

import numpy as np
from numpy.random import Generator

from htm_rl.agents.agent import Agent
from htm_rl.agents.q.eligibility_traces import EligibilityTraces
from htm_rl.agents.q.qvn import QValueNetwork
from htm_rl.agents.q.sa_encoder import SaEncoder
from htm_rl.agents.qmb.reward_model import RewardModel
from htm_rl.agents.qmb.transition_model import TransitionModel, make_s_a_transition_model
from htm_rl.common.sdr import SparseSdr
from htm_rl.common.utils import exp_decay
from htm_rl.envs.env import Env


class QModelBasedAgent(Agent):
    n_actions: int
    sa_encoder: SaEncoder
    Q: QValueNetwork
    E_traces: Optional[EligibilityTraces]

    sa_transition_model: TransitionModel
    reward_model: RewardModel

    exploration_eps: Optional[tuple[float, float]]
    softmax_enabled: bool

    _current_sa_sdr: Optional[SparseSdr]
    _rng: Generator

    def __init__(
            self,
            env: Env,
            seed: int,
            sa_encoder: dict,
            qvn: dict,
            reward_model: dict,
            transition_model: dict,
            eligibility_traces: dict = None,
            exploration_eps: tuple[float, float] = None,
            softmax_enabled: bool = False
    ):
        self.n_actions = env.n_actions
        self.sa_encoder = SaEncoder(env, seed, **sa_encoder)
        self.Q = QValueNetwork(self.sa_encoder.output_sdr_size, seed, **qvn)
        self.E_traces = EligibilityTraces(
            self.sa_encoder.output_sdr_size,
            **eligibility_traces
        )
        self.sa_transition_model = make_s_a_transition_model(
            self.sa_encoder.state_sp, self.sa_encoder.action_encoder,
            **transition_model
        )
        self.reward_model = RewardModel(self.sa_encoder.output_sdr_size, **reward_model)
        self.exploration_eps = exploration_eps
        self.softmax_enabled = softmax_enabled

        self._rng = np.random.default_rng(seed)
        self._current_sa_sdr = None

    @property
    def name(self):
        return 'qmb'

    def act(self, reward: float, state: SparseSdr, first: bool):
        if first:
            self.on_new_episode()

        s = self.sa_encoder.encode_state(state, learn=True)
        actions_sa_sdr = self.sa_encoder.encode_actions(s, learn=True)

        action_values = self.Q.values(actions_sa_sdr)
        greedy_action = np.argmax(action_values)
        if not first:
            self.reward_model.update(s, reward)
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

        self._current_sa_sdr = actions_sa_sdr[action]
        self.process_transition(s, action, learn_tm=True)
        return action

    def process_transition(self, s, action, learn_tm: bool) -> tuple[SparseSdr, SparseSdr]:
        a = self.sa_encoder.action_encoder.encode(action)
        s_a = self.sa_encoder.sa_concatenator.concatenate(s, a)
        return self.sa_transition_model.process(s_a, learn=learn_tm)

    def on_new_episode(self):
        self.Q.decay_learning_factors()
        self.E_traces.reset()
        if self.exploration_eps is not None:
            exp_decay(self.exploration_eps)
        self.sa_transition_model.reset()


def softmax(x):
    """Computes softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)
