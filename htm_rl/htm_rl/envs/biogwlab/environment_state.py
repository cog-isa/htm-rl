from copy import copy
from typing import Tuple, Optional, Dict, Union, List

import numpy as np

from htm_rl.common.plot_utils import plot_grid_image
from htm_rl.common.sdr_encoders import IntBucketEncoder, SdrConcatenator, IntArrayEncoder
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
    _render: List[str]
    _encoders: Dict
    _cached_renderings: Dict
    _encoding_sdr_concatenator: SdrConcatenator

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

    def set_rendering(self, render: List[str], **renderer):
        self._render = render
        self._encoders = dict()
        self._cached_renderings = dict()

        for data_name in render:
            if data_name == 'position':
                self._encoders['position'] = IntBucketEncoder(n_values=self.n_cells, **renderer)
            elif data_name == 'direction':
                self._encoders['direction'] = IntBucketEncoder(n_values=len(MOVE_DIRECTIONS), **renderer)
            elif data_name == 'obstacles':
                self._encoders['obstacles'] = IntArrayEncoder(shape=self.shape, n_types=1)
            elif data_name == 'food':
                self._encoders['food'] = IntArrayEncoder(shape=self.shape, n_types=1)

        if len(self._render) == 1:
            encoder = self._encoders[self._render[0]]
            self.output_sdr_size = encoder.output_sdr_size
        else:
            encoders = [self._encoders[data_name] for data_name in self._render]
            self._encoding_sdr_concatenator = SdrConcatenator(encoders)
            self.output_sdr_size = self._encoding_sdr_concatenator.output_sdr_size

    def render(self):
        observation = []
        for data_name in self._render:
            encoded_data = None
            if data_name in self._cached_renderings:
                encoded_data = self._cached_renderings[data_name]
            elif data_name == 'position':
                position_fl = self._flatten_position(self.agent_position)
                encoded_data = self._encoders[data_name].encode(position_fl)
            elif data_name == 'direction':
                encoded_data = self._encoders[data_name].encode(self.agent_view_direction)
            elif data_name == 'obstacles':
                encoded_data = self._encoders[data_name].encode(self.obstacle_mask, self.obstacle_mask)
                self._cached_renderings[data_name] = encoded_data
            elif data_name == 'food':
                encoded_data = self._encoders['food'].encode(self.food_mask, self.food_mask)

            observation.append(encoded_data)

        if len(observation) == 1:
            return observation[0]
        else:
            return self._encoding_sdr_concatenator.concatenate(*observation)

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
