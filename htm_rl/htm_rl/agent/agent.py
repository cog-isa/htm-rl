from collections import deque

import numpy as np

from htm_rl.agent.memory import Memory
from htm_rl.agent.planner import Planner
from htm_rl.common.base_sar import Sar


class Agent:
    def __init__(self, memory: Memory, planner: Planner, n_actions):
        self.memory = memory
        self.planner = planner
        self._n_actions = n_actions
        self._planned_actions = deque(maxlen=planner.max_steps)

    def reset(self):
        self.memory.tm.reset()

    def _make_action(self, state, reward, verbose):
        if not self._planned_actions:
            from_sar = Sar(state, None, reward)
            planned_actions = self.planner.plan_actions(from_sar, verbose)
            self._planned_actions.extend(planned_actions)

        if not self._planned_actions:
            self._planned_actions.append(np.random.choice(self._n_actions))

        return self._planned_actions.popleft()

    def make_step(self, state, reward, verbose):
        action = self._make_action(state, reward, verbose)
        self.memory.train(Sar(state, action, reward), verbose)
        return action
