from copy import copy
from typing import Tuple, List, Optional, Dict

import numpy as np

from htm_rl.common.utils import clip


class EnvironmentState:
    directions = {
        # (i, j)-based coordinates [or (y, x) for the viewer]
        'right': (0, 1), 'down': (1, 0), 'left': (0, -1), 'up': (-1, 0)
    }
    directions_order = ['right', 'down', 'left', 'up']
    turn_directions = {'right': 1, 'left': -1}

    seed: int
    shape: Tuple[int, int]
    n_cells: int
    obstacle_mask: np.ndarray
    food_mask: np.ndarray
    n_foods: int
    agent_position: Tuple[int, int]
    agent_view_direction: int
    episode_step: int
    step_reward: float

    food_reward: float
    action_cost: float
    action_weight: Dict[str, float]

    _obstacle_density: float

    def __init__(self, shape_xy: Tuple[int, int], seed: int):
        # convert from x,y to i,j
        width, height = shape_xy
        self.shape = (height, width)
        self.n_cells = height * width

        self.seed = seed
        self.episode_step = 0
        self.step_reward = 0

    def make_copy(self):
        env = copy(self)
        env.food_mask = self.food_mask.copy()
        return env

    def observe(self):
        reward = self.step_reward
        obs = self._flatten_position(self.agent_position)
        is_first = self.episode_step == 0
        return reward, obs, is_first

    def act(self, action: str):
        reward = 0
        if action == 'stay':
            success, reward = self.stay()
        elif action.startswith('turn '):
            turn_direction = action[5:]  # cut "turn "
            turn_direction = self.turn_directions[turn_direction]

            self.agent_view_direction, reward = self.turn(turn_direction)
        else:
            direction = action[5:]  # cut "move "
            if direction == 'forward':
                # move direction is view direction
                direction = self.directions_order[self.agent_view_direction]

            direction = self.directions[direction]
            self.agent_position, success, reward = self.move(direction)

        reward += self._collect()
        self.step_reward = reward
        self.episode_step += 1

    def _collect(self):
        i, j = self.agent_position
        reward = 0
        if self.food_mask[i, j]:
            self.food_mask[i, j] = False
            self.n_foods -= 1
            reward += self.food_reward
        return reward

    def is_terminal(self):
        no_foods = self.n_foods <= 0
        return no_foods

    def generate_obstacles(self, density: float):
        self._obstacle_density = density

        height, width = self.shape
        rng = np.random.default_rng(seed=self.seed)

        n_cells = self.n_cells
        n_required_obstacles = int((1. - self._obstacle_density) * n_cells)

        clear_path_mask = np.zeros(self.shape, dtype=np.bool)
        non_visited_neighbors = np.empty_like(clear_path_mask, dtype=np.float)

        p_change_cell = n_cells ** -.25
        p_move_forward = 1. - n_cells ** -.375

        position = self._centered_rand2d(height, width, rng)
        view_direction = rng.integers(4)
        clear_path_mask[position] = True
        n_obstacles = 0

        while n_obstacles < n_required_obstacles:
            moved_forward = False
            if rng.random() < p_move_forward:
                new_position = self._move_forward(position, view_direction)
                new_position = self._clip_position(new_position)
                if not clear_path_mask[new_position]:
                    position = new_position
                    clear_path_mask[position] = True
                    n_obstacles += 1
                    moved_forward = True

            if not moved_forward:
                view_direction = self._turn(view_direction, rng)

            if rng.random() < p_change_cell:
                position, view_direction = self._choose_rnd_cell(
                    clear_path_mask, non_visited_neighbors, rng
                )

        obstacle_mask = ~clear_path_mask
        self.obstacle_mask = obstacle_mask

    @classmethod
    def _move_forward(cls, position, view_direction):
        view_direction = cls.directions_order[view_direction]
        direction = cls.directions[view_direction]
        i, j = position

        i += direction[0]
        j += direction[1]
        return i, j

    def _clip_position(self, position):
        i, j = position
        i = clip(i, self.shape[0])
        j = clip(j, self.shape[1])
        return i, j

    def _choose_rnd_cell(
            self, gridworld: np.ndarray, non_visited_neighbors: np.ndarray,
            rnd: np.random.Generator
    ):
        # count non-visited neighbors
        non_visited_neighbors.fill(0)
        non_visited_neighbors[1:] += gridworld[1:] * (~gridworld[:-1])
        non_visited_neighbors[:-1] += gridworld[:-1] * (~gridworld[1:])
        non_visited_neighbors[:, 1:] += gridworld[:, 1:] * (~gridworld[:, :-1])
        non_visited_neighbors[:, :-1] += gridworld[:, :-1] * (~gridworld[:, 1:])
        # normalize to become probabilities
        non_visited_neighbors /= non_visited_neighbors.sum()

        # choose cell
        flatten_visited_indices = np.flatnonzero(non_visited_neighbors)
        probabilities = non_visited_neighbors.ravel()[flatten_visited_indices]
        cell_flatten_index = rnd.choice(flatten_visited_indices, p=probabilities)
        i, j = divmod(cell_flatten_index, self.shape[1])

        # choose direction
        view_direction = rnd.integers(4)
        return (i, j), view_direction

    @staticmethod
    def _centered_rand2d(max_i, max_j, rnd):
        mid_i = (max_i + 1)//2
        mid_j = (max_j + 1)//2

        i = mid_i + rnd.integers(-max_i//4, max_i//4 + 1)
        j = mid_j + rnd.integers(-max_j//4, max_j//4 + 1)
        return i, j

    def generate_food(self, reward: float):
        rnd = np.random.default_rng(seed=self.seed)
        n_cells = self.n_cells

        # n_foods = max(int(n_cells ** .5 - 2), 1)
        # print(n_foods)

        n_foods = 1

        # work in flatten then reshape
        empty_positions = np.flatnonzero(~self.obstacle_mask)
        food_positions = rnd.choice(empty_positions, size=n_foods, replace=False)

        food_mask = np.zeros(n_cells, dtype=np.bool)
        food_mask[food_positions] = True
        food_mask = food_mask.reshape(self.shape)

        self.food_mask = food_mask
        self.n_foods = n_foods
        self.food_reward = reward

    def spawn_agent(self):
        rnd = np.random.default_rng(self.seed)

        available_positions_mask = ~self.obstacle_mask
        available_positions_fl = np.flatnonzero(available_positions_mask)
        position_fl = rnd.choice(available_positions_fl)
        position = self._unflatten_position(position_fl)
        self.agent_position = position

        self.agent_view_direction = rnd.choice(4)

    def add_action_costs(self, action_cost: float, action_weight: Dict[str, float]):
        self.action_cost = action_cost
        self.action_weight = action_weight

    def _flatten_position(self, position):
        i, j = position
        return i * self.shape[1] + j

    def _unflatten_position(self, flatten_position):
        return divmod(flatten_position, self.shape[1])

    def stay(self):
        success = True
        reward = self.action_weight['stay'] * self.action_cost
        return success, reward

    def move(self, direction):
        new_position, success = self._try_move(direction)
        reward = self.action_weight['move'] * self.action_cost
        return new_position, success, reward

    def _try_move(self, move_direction):
        i, j = self.agent_position

        i += move_direction[0]
        j += move_direction[1]

        i = clip(i, self.shape[0])
        j = clip(j, self.shape[1])
        success = (i, j) != self.agent_position and not self.obstacle_mask[i, j]
        new_position = (i, j)
        if not success:
            new_position = self.agent_position
        return new_position, success

    @staticmethod
    def make(
            shape_xy: Tuple[int, int], seed: int,
            obstacle_density: float,
            move_dynamics: Dict,
            **environment
    ):
        state = EnvironmentState(shape_xy=shape_xy, seed=seed)
        state.generate_obstacles(obstacle_density)
        state.generate_food(**environment['food'])
        state.spawn_agent()
        state.add_action_costs(**move_dynamics)
        return state

    @staticmethod
    def _turn(view_direction, rnd):
        turn_direction = int(np.sign(.5 - rnd.random()))
        return (view_direction + turn_direction + 4) % 4

    def turn(self, turn_direction):
        new_direction = (self.agent_view_direction + turn_direction + 4) % 4
        reward = self.action_weight['turn'] * self.action_cost
        return new_direction, reward


class MoveDynamics:
    action_cost: float
    action_weight: Dict[str, float]

    def __init__(self, action_cost: float, action_weight: Dict[str, float]):
        self.action_cost = action_cost
        self.action_weight = action_weight

    def stay(self, state):
        success = True
        reward = self.action_weight['stay'] * self.action_cost
        return state.agent_position, success, reward

    def move(self, state, direction):
        new_position, success = self._try_move(state, direction)
        reward = self.action_weight['move'] * self.action_cost
        return new_position, success, reward

    @staticmethod
    def _try_move(state: EnvironmentState, move_direction):
        i, j = state.agent_position

        i += move_direction[0]
        j += move_direction[1]

        i = clip(i, state.shape[0])
        j = clip(j, state.shape[1])
        success = (i, j) != state.agent_position and not state.obstacle_mask[i, j]
        new_position = (i, j)
        if not success:
            new_position = state.agent_position
        return new_position, success