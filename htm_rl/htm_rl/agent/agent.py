from collections import deque

import numpy as np

from htm_rl.agent.memory import Memory
from htm_rl.agent.planner import Planner
from htm_rl.common.base_sar import Sar
from htm_rl.common.utils import trace


class Agent:
    def __init__(self, memory: Memory, planner: Planner, n_actions, use_cooldown=False):
        self.memory = memory
        self.planner = planner
        self._n_actions = n_actions
        self.set_planning_horizon(planner.max_steps)
        self._use_cooldown = use_cooldown

    def set_planning_horizon(self, planning_horizon):
        self.planner.max_steps = planning_horizon
        self._planning_horizon = planning_horizon

        self._planning_enabled = planning_horizon > 0
        self._n_times_planned = 0
        self._n_times_random = 0
        self._init_planning()
        self._cooldown_scaler = self._planning_horizon / .2

    def reset(self):
        self.memory.tm.reset()
        self._init_planning()

    def make_step(self, state, reward, verbose):
        trace(verbose, f'\nState: {state}; reward: {reward}')
        action = self._make_action(state, reward, verbose)
        trace(verbose, f'\nMake action: {action}')

        self.memory.train(Sar(state, action, reward), verbose)
        return action

    def _init_planning(self):
        self._planned_actions = deque(maxlen=self._planning_horizon + 1)
        self._planning_cooldown = 0

    def _make_action(self, state, reward, verbose):
        if self._should_plan:
            from_sar = Sar(state, None, reward)
            planned_actions = self.planner.plan_actions(from_sar, verbose)
            self._planned_actions.extend(planned_actions)
            self._raise_cooldown()
            self._n_times_planned += 1

        if not self._planned_actions:
            self._planned_actions.append(np.random.choice(self._n_actions))
            self._decrease_cooldown()
            self._n_times_random += 1

        return self._planned_actions.popleft()

    @property
    def _should_plan(self):
        enabled, planned = self._planning_enabled, self._planned_actions
        on_cooldown = self._planner_on_cooldown
        return enabled and not planned and not on_cooldown

    @property
    def _planner_on_cooldown(self):
        return self._planning_cooldown > 0

    def _raise_cooldown(self):
        if not self._planned_actions and self._use_cooldown:
            self._planning_cooldown = round(np.random.rand() * self._cooldown_scaler)
            self._planning_cooldown += 1

    def _decrease_cooldown(self):
        if self._planning_enabled and self._use_cooldown:
            self._planning_cooldown -= 1

    @property
    def plan_to_random_ratio(self):
        return (self._n_times_planned + 1) / (self._n_times_random + 1)
