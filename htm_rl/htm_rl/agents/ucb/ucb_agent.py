from typing import Dict

from htm_rl.agents.ucb.ucb_actor_critic import UcbActorCritic
from htm_rl.common.sdr import SparseSdr
from htm_rl.common.sdr_encoders import IntBucketEncoder, SdrConcatenator
from htm_rl.common.utils import timed
from htm_rl.envs.biogwlab.environment import BioGwLabEnvironment
from htm_rl.htm_plugins.spatial_pooler import SpatialPooler


class UcbAgent:
    _n_actions: int

    _state_sp: SpatialPooler
    _action_encoder: IntBucketEncoder
    _sa_concatenator: SdrConcatenator
    _sa_sp: SpatialPooler

    _ucb_actor_critic: UcbActorCritic

    def __init__(
            self,
            env: BioGwLabEnvironment,
            ucb_actor_critic: Dict,
            state_sp: Dict,
            action_encoder: Dict,
            sa_sp: Dict
    ):
        self._state_sp = SpatialPooler(input_source=env, **state_sp)
        self._action_encoder = IntBucketEncoder(n_values=env.n_actions, **action_encoder)
        self._sa_concatenator = SdrConcatenator(input_sources=[
            self._state_sp,
            self._action_encoder
        ])
        self._sa_sp = SpatialPooler(input_source=self._sa_concatenator, **sa_sp)

        self._ucb_actor_critic = UcbActorCritic(
            cells_sdr_size=self._sa_sp.output_sdr_size,
            **ucb_actor_critic
        )
        self._n_actions = env.n_actions

    @timed
    def run_episode(self, env):
        self._ucb_actor_critic.reset()
        step = 0
        total_reward = 0.

        reward, state, first = env.observe()
        while not (first and step > 0):
            action = self.choose_action(state)
            env.act(action)

            reward, new_state, first = env.observe()
            self.get_feedback(state, action, reward, new_state)

            state = new_state
            step += 1
            total_reward += reward

        return step, total_reward

    def choose_action(self, state):
        actions = self.encode_actions(state)
        action = self._ucb_actor_critic.choose(actions)
        return action

    def get_feedback(self, state, action, reward, new_state):
        sa_sdr = self.encode_sa(state, action, learn=True)
        self._ucb_actor_critic.add_step(sa_sdr, reward)

    def encode_actions(self, state: SparseSdr):
        actions = []

        for action in range(self._n_actions):
            sa_sdr = self.encode_sa(state, action, learn=False)
            actions.append(sa_sdr)

        return actions

    def encode_sa(self, state: SparseSdr, action: int, learn: bool) -> SparseSdr:
        s = self._state_sp.compute(state, learn=learn)
        a = self._action_encoder.encode(action)

        sa_concat_sdr = self._sa_concatenator.concatenate(s, a)
        sa_sdr = self._sa_sp.compute(sa_concat_sdr, learn=learn)
        return sa_sdr
