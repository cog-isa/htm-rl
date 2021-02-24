import numpy as np
from numpy.random._generator import Generator

from htm_rl.common.utils import isnone
from htm_rl.envs.biogwlab.dynamics import BioGwLabEnvDynamics
from htm_rl.envs.biogwlab.environment_state import BioGwLabEnvState


class BioGwLabEnv:
    rnd: Generator
    actions = ['stay', 'move', 'turn left', 'turn right']

    state: BioGwLabEnvState
    dynamics: BioGwLabEnvDynamics

    def __init__(self, state: BioGwLabEnvState, dynamics: BioGwLabEnvDynamics):
        self.state = state
        self.dynamics = dynamics
        self.rnd = np.random.default_rng(seed=state.seed)

        self.generate_initial_position()

    def generate_initial_position(self):
        position = self._get_random_empty_position()
        direction = self.rnd.integers(len(self.state.directions))

        self.state.agent_initial_position = position
        self.state.agent_initial_direction = direction

    def reset(self):
        """ reset the game, return the initial state"""
        self._reset_food()

        self.state.agent_position = self.state.agent_initial_position
        self.state.agent_direction = self.state.agent_initial_direction
        return self._get_agent_state()

    def is_terminal(self, state=None):
        """ return True if position is terminal or False if it isn't """
        state = isnone(state, self.state)
        return self.dynamics.is_terminal(state)

    def step(self, action):
        """ take action, return next_state, reward, is_done, empty_info """
        assert not self.is_terminal(), 'Episode is finished!'

        state = self.state
        action = self.actions[action]
        reward = 0
        if action == 'stay':
            reward += self.dynamics.stay(state)
        elif action == 'move':
            reward += self.dynamics.move_forward(state)
            reward += self.dynamics.stay(state)
        elif action == 'turn left':
            reward += self.dynamics.turn(-1, state)
            reward += self.dynamics.stay(state)
        elif action == 'turn right':
            reward += self.dynamics.turn(1, state)
            reward += self.dynamics.stay(state)
        else:
            raise NotImplementedError(f'Action {action} is not implemented')

        agent_state = self._get_agent_state()
        is_done = self.is_terminal()
        return agent_state, reward, is_done, {}

    def _reset_food(self):
        # print('<=', self.state.n_rewarding_foods)
        self.state.n_rewarding_foods = self.dynamics.count_rewarding_food_items(self.state)
        # print('=>', self.state.n_rewarding_foods)
        for i, j, food_type in self.state.food_items:
            self.state.food_mask[i, j] = True
            self.state.food_map[i, j] = food_type

    def _get_agent_state(self):
        position = self.state.agent_position
        direction = self.state.directions[self.state.agent_direction]
        return position, direction

    def _unflatten(self, flatten_position):
        return divmod(flatten_position, self.state.size)

    def _get_random_empty_position(self):
        non_empty_mask = self.state.obstacle_mask | self.state.food_mask
        available_states = np.flatnonzero(non_empty_mask)
        position = self.rnd.choice(available_states)
        return self._unflatten(position)