from copy import copy
from typing import Tuple, Optional, Dict, Union

import numpy as np

from htm_rl.common.plot_utils import plot_grid_image
from htm_rl.common.utils import isnone
from htm_rl.envs.biogwlab.agent_position_encoder import AgentPositionEncoder
from htm_rl.envs.biogwlab.generation.obstacles import ObstacleGenerator
from htm_rl.envs.biogwlab.move_dynamics import MOVE_DIRECTIONS, DIRECTIONS_ORDER, TURN_DIRECTIONS, MoveDynamics


class EnvironmentState:
    supported_actions = [
        'stay',
        'move left', 'move up', 'move right', 'move down',
        'move forward', 'turn left', 'turn right'
    ]

    seed: int
    shape: Tuple[int, int]
    n_cells: int
    obstacle_mask: np.ndarray

    food_mask: np.ndarray
    n_foods: int
    agent_position: Tuple[int, int]
    agent_view_direction: Optional[int]
    episode_step: int
    step_reward: float

    food_reward: float
    action_cost: float
    action_weight: Dict[str, float]

    output_sdr_size: int

    _obstacle_generator: ObstacleGenerator
    _render_mode: str
    _renderer: Union['AgentPositionEncoder']

    def __init__(
            self, shape_xy: Tuple[int, int], seed: int,
    ):
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
        obs = self.render()
        is_first = self.episode_step == 0
        return reward, obs, is_first

    def act(self, action: str):
        self.step_reward = 0
        if action == 'stay':
            self.stay()
        elif action.startswith('turn '):
            turn_direction = action[5:]  # cut "turn "
            turn_direction = TURN_DIRECTIONS[turn_direction]

            self.turn(turn_direction)
        else:
            direction = action[5:]  # cut "move "
            if direction == 'forward':
                # move direction is view direction
                direction = DIRECTIONS_ORDER[self.agent_view_direction]

            direction = MOVE_DIRECTIONS[direction]
            self.move(direction)

        self.collect()
        self.episode_step += 1

    def stay(self):
        self.step_reward += self.action_weight['stay'] * self.action_cost

    def move(self, direction):
        self.agent_position, success = MoveDynamics.try_move(
            self.agent_position, direction, self.shape, self.obstacle_mask
        )
        self.step_reward += self.action_weight['move'] * self.action_cost

    def turn(self, turn_direction):
        self.agent_view_direction = MoveDynamics.turn(self.agent_view_direction, turn_direction)
        self.step_reward = self.action_weight['turn'] * self.action_cost

    def collect(self):
        if self.food_mask[self.agent_position]:
            self.food_mask[self.agent_position] = False
            self.n_foods -= 1
            self.step_reward += self.food_reward

    def is_terminal(self):
        no_foods = self.n_foods <= 0
        return no_foods

    def set_obstacle_generator(self, obstacle_density: float):
        self._obstacle_generator = ObstacleGenerator(
            obstacle_density=obstacle_density, shape=self.shape, seed=self.seed
        )

    def generate_obstacles(self):
        self.obstacle_mask = self._obstacle_generator.generate()

    def _flatten_position(self, position):
        i, j = position
        return i * self.shape[1] + j

    def _unflatten_position(self, flatten_position):
        return divmod(flatten_position, self.shape[1])

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
        rnd.integers(1000, size=5)

        available_positions_mask = ~self.obstacle_mask
        available_positions_fl = np.flatnonzero(available_positions_mask)
        position_fl = rnd.choice(available_positions_fl)
        position = self._unflatten_position(position_fl)
        self.agent_position = position
        self.agent_view_direction = rnd.choice(4)

    def add_action_costs(self, action_cost: float, action_weight: Dict[str, float]):
        self.action_cost = action_cost
        self.action_weight = action_weight

    def set_rendering(self, render_mode=None, include_view_direction=False, **renderer):
        self._render_mode = isnone(render_mode, "agent")

        if render_mode == "agent":
            self._renderer = AgentPositionEncoder(
                n_positions=self.n_cells, n_directions=len(MOVE_DIRECTIONS),
                include_view_direction=include_view_direction,
                **renderer
            )
            self.output_sdr_size = self._renderer.output_sdr_size

    def render(self):
        if self._render_mode == "agent":
            position_fl = self._flatten_position(self.agent_position)
            state = (position_fl, self.agent_view_direction)
            return self._renderer.encode(state)

    def render_rgb(self):
        img = np.zeros(self.shape, dtype=np.int8)
        img[self.obstacle_mask] = -2
        img[self.food_mask] = 2
        img[self.agent_position] = 4
        plot_grid_image(img)

    @staticmethod
    def make(
            shape_xy: Tuple[int, int], seed: int,
            obstacle_density: float,
            action_costs: Dict,
            **environment
    ):
        state = EnvironmentState(
            shape_xy=shape_xy, seed=seed
        )
        state.add_action_costs(**action_costs)
        state.set_obstacle_generator(obstacle_density=obstacle_density)

        state.generate_obstacles()
        state.generate_food(**environment['food'])

        state.set_rendering(**environment['rendering'])
        state.spawn_agent()
        return state
