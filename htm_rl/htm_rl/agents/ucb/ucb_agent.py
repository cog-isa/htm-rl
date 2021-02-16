from htm_rl.agents.ucb.processing_unit import ConcatenateUnit
from htm_rl.agents.ucb.ucb_actor_critic import UcbActorCritic
from htm_rl.common.base_sa import Sa
from htm_rl.common.sa_sdr_encoder import SaSdrEncoder
from htm_rl.common.sdr import SparseSdr
from htm_rl.common.ucb_sa_sdr_encoder import UcbSaSdrEncoder
from htm_rl.common.utils import trace, timed
from htm_rl.htm_plugins.spatial_pooler import SpatialPooler
from htm_rl.htm_plugins.ucb_spatial_pooler import UcbSpatialPooler


class UcbAgent:
    _ucb_actor_critic: UcbActorCritic
    _n_actions: int

    encoder: UcbSaSdrEncoder
    spatial_pooler: UcbSpatialPooler

    def __init__(
            self, ucb_actor_critic,
            encoder: UcbSaSdrEncoder, spatial_pooler: UcbSpatialPooler,
            n_actions: int
    ):
        self.encoder = encoder
        self.spatial_pooler = spatial_pooler
        self._ucb_actor_critic = UcbActorCritic(
            cells_sdr_size=spatial_pooler.output_shape[0],
            **ucb_actor_critic
        )
        self._n_actions = n_actions

    @timed
    def run_episode(self, env, max_steps, verbosity):
        self.reset()
        state, reward, done = env.reset(), 0, env.is_terminal()
        action = self.choose_action(state, reward, done, verbosity)

        step = 0
        total_reward = 0.
        while step < max_steps and not done:
            state, reward, done, info = env.step(action)
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
        sa = Sa(state, action)
        sa_sdr = self.encode_sa(sa, learn=True)
        self._ucb_actor_critic.add_step(sa_sdr, reward)
        return action

    def _make_action(self, state, verbosity: int):
        current_sa = Sa(state, None)
        options = self.predict_states(current_sa, verbosity)
        action = self._ucb_actor_critic.choose(options)

        return action

    def encode_sa(self, sa: Sa, learn: bool) -> SparseSdr:
        sa_sdr = self.encoder.process(sa)
        sa_sdr = self.spatial_pooler.encode(sa_sdr, learn=learn)
        return sa_sdr

    def predict_states(self, initial_sa: Sa, verbosity: int):
        action_outcomes = []
        trace(verbosity, 2, '\n======> Planning')

        state = initial_sa.state
        for action in range(self._n_actions):
            sa = Sa(state, action)
            sa_sdr = self.encode_sa(sa, learn=False)
            action_outcomes.append(sa_sdr)

        trace(verbosity, 2, '<====== Planning complete')
        return action_outcomes
