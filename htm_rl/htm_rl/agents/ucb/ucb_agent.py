from htm_rl.agents.ucb.ucb_actor_critic import UcbActorCritic
from htm_rl.agents.ucb.ucb_planner import UcbPlanner
from htm_rl.common.base_sa import Sa
from htm_rl.common.utils import trace, timed


class UcbAgent:
    planner: UcbPlanner
    _ucb_actor_critic: UcbActorCritic
    _n_actions: int

    def __init__(
            self, planner: UcbPlanner,
            ucb_actor_critic: UcbActorCritic, n_actions
    ):
        self.planner = planner
        self._ucb_actor_critic = ucb_actor_critic
        self._n_actions = n_actions

    @timed
    def run_episode(self, env, max_steps, verbosity):
        self.reset()
        state, reward, done = env.reset(), 0, env.is_terminal()
        action = self.make_step(state, reward, done, verbosity)

        step = 0
        total_reward = 0.
        while step < max_steps and not done:
            state, reward, done, info = env.step(action)
            action = self.make_step(state, reward, done, verbosity)
            step += 1
            total_reward += reward

        return step, total_reward

    def reset(self):
        self._ucb_actor_critic.reset()

    def make_step(self, state, reward, is_done, verbosity: int):
        trace(verbosity, 2, f'\nState: {state}; reward: {reward}')

        action = self._make_action(state, verbosity)
        trace(verbosity, 2, f'\nMake action: {action}')

        # learn
        sa = Sa(state, action)
        sa_sdr = self.planner.encode_sa(sa, learn=True)
        self._ucb_actor_critic.add_step(sa_sdr, reward)
        return action

    def _make_action(self, state, verbosity: int):
        current_sa = Sa(state, None)
        options = self.planner.predict_states(current_sa, verbosity)
        action = self._ucb_actor_critic.choose(options)

        return action
