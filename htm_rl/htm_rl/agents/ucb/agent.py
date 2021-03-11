from typing import Dict, Optional, List

from htm_rl.agents.agent import Agent
from htm_rl.agents.ucb.ucb_actor_critic import UcbActorCritic
from htm_rl.common.sdr import SparseSdr
from htm_rl.common.sdr_encoders import IntBucketEncoder, SdrConcatenator
from htm_rl.envs.env import Env
from htm_rl.htm_plugins.spatial_pooler import SpatialPooler


class UcbAgent(Agent):
    _n_actions: int
    _current_sa_sdr: Optional[SparseSdr]

    _state_sp: SpatialPooler
    _action_encoder: IntBucketEncoder
    _sa_concatenator: SdrConcatenator
    _sa_sp: SpatialPooler

    _ucb_actor_critic: UcbActorCritic

    def __init__(
            self,
            env: Env,
            seed: int,
            ucb_actor_critic: Dict,
            state_sp: Dict,
            action_encoder: Dict,
            sa_sp: Dict
    ):
        self._state_sp = SpatialPooler(input_source=env, seed=seed, **state_sp)
        self._action_encoder = IntBucketEncoder(n_values=env.n_actions, **action_encoder)
        self._sa_concatenator = SdrConcatenator(input_sources=[
            self._state_sp,
            self._action_encoder
        ])
        self._sa_sp = SpatialPooler(input_source=self._sa_concatenator, seed=seed, **sa_sp)

        self._ucb_actor_critic = UcbActorCritic(
            cells_sdr_size=self._sa_sp.output_sdr_size,
            seed=seed,
            **ucb_actor_critic
        )
        self._n_actions = env.n_actions

        self._current_sa_sdr = None

    @property
    def name(self):
        return 'ucb'

    def act(self, reward: float, state: SparseSdr, first: bool):
        if not first:
            # process feedback
            state, action = self._current_sa_sdr
            self._current_sa_sdr = self._encode_sa(state, action, learn=True)
            self._ucb_actor_critic.add_step(self._current_sa_sdr, reward)

        actions = self._encode_actions(state)
        action = self._ucb_actor_critic.choose(actions)
        # self._current_sa_sdr = actions[action]
        self._current_sa_sdr = state, action

        return action

    def _encode_actions(self, state: SparseSdr) -> List[SparseSdr]:
        actions = []

        for action in range(self._n_actions):
            sa_sdr = self._encode_sa(state, action, learn=False)
            actions.append(sa_sdr)

        return actions

    def _encode_sa(self, state: SparseSdr, action: int, learn: bool) -> SparseSdr:
        s = self._state_sp.compute(state, learn=learn)
        a = self._action_encoder.encode(action)

        sa_concat_sdr = self._sa_concatenator.concatenate(s, a)
        sa_sdr = self._sa_sp.compute(sa_concat_sdr, learn=learn)
        return sa_sdr
