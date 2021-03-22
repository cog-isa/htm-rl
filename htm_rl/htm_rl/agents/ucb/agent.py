from typing import Dict, Optional, List

from htm_rl.agents.agent import Agent
from htm_rl.agents.ucb.sparse_value_network import SparseValueNetwork
from htm_rl.common.sdr import SparseSdr
from htm_rl.common.sdr_encoders import IntBucketEncoder, SdrConcatenator
from htm_rl.envs.env import Env
from htm_rl.htm_plugins.spatial_pooler import SpatialPooler


class UcbAgent(Agent):
    _n_actions: int
    _current_sa: Optional[SparseSdr]
    _td_lambda_learning: bool

    state_sp: SpatialPooler
    action_encoder: IntBucketEncoder
    sa_concatenator: SdrConcatenator
    sa_sp: SpatialPooler

    q_network: SparseValueNetwork

    def __init__(
            self,
            env: Env,
            seed: int,
            q_network: Dict,
            state_sp: Dict,
            action_encoder: Dict,
            sa_sp: Dict
    ):
        self.state_sp = SpatialPooler(input_source=env, seed=seed, **state_sp)
        self.action_encoder = IntBucketEncoder(n_values=env.n_actions, **action_encoder)
        self.sa_concatenator = SdrConcatenator(input_sources=[
            self.state_sp,
            self.action_encoder
        ])
        self.sa_sp = SpatialPooler(input_source=self.sa_concatenator, seed=seed, **sa_sp)

        self.q_network = SparseValueNetwork(
            cells_sdr_size=self.sa_sp.output_sdr_size,
            seed=seed,
            **q_network
        )
        self._n_actions = env.n_actions
        self._current_sa = None
        self._td_lambda_learning = True

    @property
    def name(self):
        return 'ucb'

    def act(self, reward: float, state: SparseSdr, first: bool):
        if first:
            self.q_network.reset()

        s = self.state_sp.compute(state, learn=True)
        return self._act(reward, s, first)

    def _act(self, reward: float, s: SparseSdr, first: bool):
        actions = self._encode_actions(s)
        action = self.q_network.choose(actions)

        if not first:
            # process feedback
            prev_sa_sdr = self._current_sa
            greedy_action = self.q_network.choose(actions, greedy=True)
            greedy_sa_sdr = actions[greedy_action]

            self.q_network.update(
                sa=prev_sa_sdr,
                reward=reward,
                sa_next=greedy_sa_sdr,
                td_lambda=self._td_lambda_learning
            )

        self._current_sa = actions[action]
        return action

    def _encode_actions(self, s: SparseSdr) -> List[SparseSdr]:
        actions = []
        for action in range(self._n_actions):
            sa_sdr = self._encode_sa(s, action, learn=True)
            actions.append(sa_sdr)

        return actions

    def _encode_sa(self, s: SparseSdr, action: int, learn: bool) -> SparseSdr:
        a = self.action_encoder.encode(action)
        s_a = self.sa_concatenator.concatenate(s, a)
        sa = self.sa_sp.compute(s_a, learn=learn)
        return sa
