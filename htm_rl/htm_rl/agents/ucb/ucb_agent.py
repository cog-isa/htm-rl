from typing import Dict

from htm_rl.agents.ucb.ucb_actor_critic import UcbActorCritic
from htm_rl.common.sdr import SparseSdr
from htm_rl.common.sdr_encoders import IntBucketEncoder, SdrConcatenator
from htm_rl.common.utils import trace, timed
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
            self._state_sp, self._action_encoder
        ])
        self._sa_sp = SpatialPooler(input_source=self._sa_concatenator, **sa_sp)

        self._ucb_actor_critic = UcbActorCritic(
            cells_sdr_size=self._sa_sp.output_sdr_size,
            **ucb_actor_critic
        )
        self._n_actions = env.n_actions

    @timed
    def run_episode(self, env, verbosity):
        self.reset()
        state, reward, done = env.reset(), 0, env.is_terminal()
        action = self.choose_action(state, reward, done, verbosity)

        step = 0
        total_reward = 0.
        while not done:
            state, reward, done, info = env.act(action)
            action = self.choose_action(state, reward, done, verbosity)
            step += 1
            total_reward += reward

        return step, total_reward

    def reset(self):
        self._ucb_actor_critic.reset()

    def choose_action(self, state, reward, is_done, verbosity: int):
        trace(verbosity, 2, f'\nState: {state}; reward: {reward}')

        action = self._make_action(state, verbosity)
        trace(verbosity, 2, f'\nMake action: {action}')

        # learn
        sa_sdr = self.encode_sa(state, action, learn=True)
        self._ucb_actor_critic.add_step(sa_sdr, reward)
        return action

    def _make_action(self, state, verbosity: int):
        options = self.predict_states(state, verbosity)
        action = self._ucb_actor_critic.choose(options)

        return action

    def encode_sa(self, state: SparseSdr, action: int, learn: bool) -> SparseSdr:
        s = self._state_sp.compute(state, learn=learn)
        # s = state
        a = self._action_encoder.encode(action)

        sa_concat_sdr = self._sa_concatenator.concatenate(s, a)
        sa_sdr = self._sa_sp.compute(sa_concat_sdr, learn=learn)
        return sa_sdr

    def predict_states(self, state, verbosity: int):
        action_outcomes = []
        trace(verbosity, 2, '\n======> Planning')

        for action in range(self._n_actions):
            sa_sdr = self.encode_sa(state, action, learn=False)
            action_outcomes.append(sa_sdr)

        trace(verbosity, 2, '<====== Planning complete')
        return action_outcomes
