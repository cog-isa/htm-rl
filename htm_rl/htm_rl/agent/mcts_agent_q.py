import numpy as np

from htm_rl.agent.mcts_actor_critic import MctsActorCritic
from htm_rl.agent.mcts_planner import MctsPlanner
from htm_rl.agent.mcts_planner_q import MctsPlannerQ
from htm_rl.common.base_sa import Sa
from htm_rl.common.utils import trace


class MctsAgentQ:
    planner: MctsPlannerQ
    _n_actions: int
    _planning_horizon: int

    def __init__(
            self, planner: MctsPlannerQ,
            mcts_actor_critic: MctsActorCritic, n_actions
    ):
        self.planner = planner
        self._mcts_actor_critic = mcts_actor_critic
        self._n_actions = n_actions

    @property
    def _planning_enabled(self):
        return self.planner.planning_horizon > 0

    def set_planning_horizon(self, planning_horizon: int):
        self.planner.planning_horizon = planning_horizon

    def reset(self):
        self._mcts_actor_critic.reset()

    def make_step(self, state, reward, is_done, verbosity: int):
        trace(verbosity, 2, f'\nState: {state}; reward: {reward}')

        action = self._make_action(state, verbosity)
        trace(verbosity, 2, f'\nMake action: {action}')

        # learn
        sa = Sa(state, action)
        sa_sdr = self.planner.encode_sa(sa, learn=True)
        self._mcts_actor_critic.add_step(sa_sdr, max(reward, 0))
        return action

    def _make_action(self, state, verbosity: int):
        if self._planning_enabled:
            current_sa = Sa(state, None)
            options = self.planner.predict_states(current_sa, verbosity)
            action = self._mcts_actor_critic.choose(options)
        else:
            action = np.random.choice(self._n_actions)
            trace(3, 1, 'RANDOM')

        return action

    def _trace_mcts_stats(self):
        ac = self._mcts_actor_critic
        q = np.round(ac.cell_value / ac.cell_visited_count, 2)
        value_bit_buckets = (q[i: i + 8] for i in range(0, q.size, 8))
        q_str = '\n'.join(
            ' '.join(map(str, value_bits)) for value_bits in value_bit_buckets
        )
        trace(2, 1, q_str)
        trace(2, 1, '=' * 20)
