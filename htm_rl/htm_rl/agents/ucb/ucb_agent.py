from htm_rl.agents.ucb.ucb_actor_critic import UcbActorCritic
from htm_rl.common.sdr import SparseSdr
from htm_rl.common.ucb_encoders import UcbIntBucketEncoder, UcbSdrConcatenator
from htm_rl.common.utils import trace, timed
from htm_rl.htm_plugins.ucb_spatial_pooler import UcbSpatialPooler


class UcbAgent:
    state_sp: UcbSpatialPooler
    action_encoder: UcbIntBucketEncoder
    sa_concat: UcbSdrConcatenator
    sa_sp: UcbSpatialPooler

    _ucb_actor_critic: UcbActorCritic
    _n_actions: int

    def __init__(
            self, ucb_actor_critic,
            state_sp: UcbSpatialPooler,
            action_encoder: UcbIntBucketEncoder,
            sa_concat: UcbSdrConcatenator,
            sa_sp: UcbSpatialPooler,
            n_actions: int
    ):
        self.state_sp = state_sp
        self.action_encoder = action_encoder
        self.sa_concat = sa_concat
        self.sa_sp = sa_sp

        self._ucb_actor_critic = UcbActorCritic(
            cells_sdr_size=sa_sp.output_sdr_size,
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
        sa_sdr = self.encode_sa(state, action, learn=True)
        self._ucb_actor_critic.add_step(sa_sdr, reward)
        return action

    def _make_action(self, state, verbosity: int):
        options = self.predict_states(state, verbosity)
        action = self._ucb_actor_critic.choose(options)

        return action

    def encode_sa(self, state: SparseSdr, action: int, learn: bool) -> SparseSdr:
        s = self.state_sp.compute(state, learn=learn)
        a = self.action_encoder.encode(action)

        sa_concat_sdr = self.sa_concat.concatenate(s, a)
        sa_sdr = self.sa_sp.compute(sa_concat_sdr, learn=learn)
        return sa_sdr

    def predict_states(self, state, verbosity: int):
        action_outcomes = []
        trace(verbosity, 2, '\n======> Planning')

        for action in range(self._n_actions):
            sa_sdr = self.encode_sa(state, action, learn=False)
            action_outcomes.append(sa_sdr)

        trace(verbosity, 2, '<====== Planning complete')
        return action_outcomes
