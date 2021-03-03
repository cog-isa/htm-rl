from copy import copy
from typing import Tuple, Optional, Dict, Union, List

import numpy as np

from htm_rl.common.plot_utils import plot_grid_images
from htm_rl.common.sdr_encoders import IntBucketEncoder, SdrConcatenator, IntArrayEncoder
from htm_rl.common.utils import isnone
from htm_rl.envs.biogwlab.generation.areas import AreasGenerator, Areas
from htm_rl.envs.biogwlab.generation.food import FoodGenerator
from htm_rl.envs.biogwlab.generation.obstacles import ObstacleGenerator, Obstacles
from htm_rl.envs.biogwlab.move_dynamics import MOVE_DIRECTIONS, DIRECTIONS_ORDER, TURN_DIRECTIONS, MoveDynamics
from htm_rl.envs.biogwlab.view_clipper import ViewClipper


class EnvironmentState:
    supported_actions = [
        'stay',
        'move left', 'move up', 'move right', 'move down',
        'move forward', 'turn left', 'turn right'
    ]

    seed: int
    shape: Tuple[int, int]
    n_cells: int
    areas: Areas
    obstacles: Obstacles

    food_mask: np.ndarray
    n_items: int
    n_foods_to_find: int
    agent_position: Tuple[int, int]
    agent_view_direction: Optional[int]
    episode_step: int
    step_reward: float

    food_rewards: Union[float, List[float]]
    action_cost: float
    action_weight: Dict[str, float]

    output_sdr_size: int

    _obstacle_generator: ObstacleGenerator
    _areas_generator: AreasGenerator

    _render: List[str]
    _encoders: Dict
    _cached_renderings: Dict
    _encoding_sdr_concatenator: SdrConcatenator
    _view_clipper: ViewClipper

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
        else:   # "move X"
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
            self.agent_position, direction, self.shape, self.obstacles.mask
        )
        self.step_reward += self.action_weight['move'] * self.action_cost

    def turn(self, turn_direction):
        self.agent_view_direction = MoveDynamics.turn(self.agent_view_direction, turn_direction)
        self.step_reward = self.action_weight['turn'] * self.action_cost

    def collect(self):
        if self.food_mask[self.agent_position]:
            self.food_mask[self.agent_position] = False
            if self.n_food_types == 1:
                reward = self.food_rewards[0]
            else:
                reward = self.food_rewards[self.food_map[self.agent_position]]

            if reward > 0:
                self.n_foods_to_find -= 1

            self.step_reward += reward

    def is_terminal(self):
        no_foods = self.n_foods_to_find <= 0
        return no_foods

    def set_obstacles(self, **generator):
        self.obstacles = Obstacles(shape=self.shape, seed=self.seed, **generator)

    def generate_obstacles(self):
        self.obstacles.generate()

    def _flatten_position(self, position):
        i, j = position
        return i * self.shape[1] + j

    def _unflatten_position(self, flatten_position):
        return divmod(flatten_position, self.shape[1])

    def generate_food(
            self, n_types: int = None, reward: int = None, rewards: List[float] = None,
            food_types: List[int] = None, n_items=None,
            n_foods_to_find = None
    ):
        food_types = isnone(food_types, [0])
        self.n_food_types = isnone(n_types, len(food_types))
        self.food_rewards = isnone(rewards, [reward])

        if self.n_food_types == 1:
            rng = np.random.default_rng(seed=self.seed)
            n_cells = self.n_cells

            # work in flatten then reshape
            empty_positions = np.flatnonzero(~self.obstacles.mask)
            food_positions = rng.choice(empty_positions, size=n_items, replace=False)

            food_mask = np.zeros(n_cells, dtype=np.bool)
            food_mask[food_positions] = True
            food_mask = food_mask.reshape(self.shape)

            self.food_mask = food_mask
            self.n_items = n_items
            n_positive_foods = n_items
        else:
            food_generator = FoodGenerator(food_types=food_types)
            self.food_items, self.food_map, self.food_mask, self.n_food_types = food_generator.generate(
                areas_map=self.areas.map,
                obstacle_mask=self.obstacles.mask,
                seed=self.seed,
                n_foods=n_items
            )
            self.n_items = len(self.food_items)
            n_positive_foods = np.sum([
                1
                for _, _, food_type in self.food_items
                if self.food_rewards[food_type] > 0
            ])

        self.n_foods_to_find = isnone(n_foods_to_find, (n_positive_foods - 1) // 3 + 1)

    def spawn_agent(self):
        rnd = np.random.default_rng(self.seed)
        # HACK: to prevent spawning in food pos if there's just 1 food item of 1 type
        rnd.integers(1000, size=5)

        available_positions_mask = ~self.obstacles.mask
        available_positions_fl = np.flatnonzero(available_positions_mask)
        position_fl = rnd.choice(available_positions_fl)
        position = self._unflatten_position(position_fl)
        self.agent_position = position
        self.agent_view_direction = rnd.choice(4)

    def add_action_costs(self, action_cost: float, action_weight: Dict[str, float]):
        self.action_cost = action_cost
        self.action_weight = action_weight

    def set_rendering(
            self, render: List[str],
            view_rectangle: Tuple = None,
            **renderer
    ):
        self._render = render
        self._encoders = dict()
        self._cached_renderings = dict()

        view_shape = self.shape
        self._view_clipper = None
        if view_rectangle is not None:
            self._view_clipper = ViewClipper(self.shape, view_rectangle)
            view_shape = self._view_clipper.view_shape

        for data_name in render:
            if data_name == 'position':
                encoder = IntBucketEncoder(n_values=self.n_cells, **renderer)
            elif data_name == 'direction':
                encoder = IntBucketEncoder(n_values=len(MOVE_DIRECTIONS), **renderer)
            elif data_name == 'obstacles':
                encoder = self.obstacles.set_renderer(view_shape)
            elif data_name == 'food':
                encoder = IntArrayEncoder(shape=view_shape, n_types=self.n_food_types)
            elif data_name == 'area':
                encoder = self.areas.set_renderer(view_shape)
            self._encoders[data_name] = encoder

        if len(self._render) == 1:
            encoder = self._encoders[self._render[0]]
            self.output_sdr_size = encoder.output_sdr_size
        else:
            encoders = [self._encoders[data_name] for data_name in self._render]
            self._encoding_sdr_concatenator = SdrConcatenator(encoders)
            self.output_sdr_size = self._encoding_sdr_concatenator.output_sdr_size

    def render(self):
        observation = []

        view_clip = None
        if self._view_clipper is not None:
            abs_indices, view_indices = self._view_clipper.clip(
                self.agent_position, self.agent_view_direction
            )
            # print(abs_indices)
            # print(view_indices)
            abs_indices = abs_indices.flatten()
            view_indices = view_indices.flatten()
            view_clip = view_indices, abs_indices

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
                encoded_data = self.obstacles.render(view_clip)
            elif data_name == 'food':
                if self._view_clipper is not None:
                    food_mask = np.ones(self._view_clipper.view_shape, dtype=np.bool).flatten()
                    food_mask[view_indices] = self.food_mask.flatten()[abs_indices]

                    food_map = np.zeros(self._view_clipper.view_shape, dtype=np.int).flatten()
                    _food_map = ~self.food_mask if self.n_food_types == 1 else self.food_map
                    food_map[view_indices] = _food_map.flatten()[abs_indices]
                    encoded_data = self._encoders[data_name].encode(food_map, food_mask)
                else:
                    encoded_data = self._encoders[data_name].encode(self.food_mask, self.food_mask)
            elif data_name == 'area':
                encoded_data = self.areas.render(view_clip)

            if encoded_data is not None:
                observation.append(encoded_data)

        if len(observation) == 1:
            return observation[0]
        else:
            return self._encoding_sdr_concatenator.concatenate(*observation)

    def render_rgb(self):
        img = np.zeros(self.shape, dtype=np.int8)
        img[self.obstacles.mask] = -2
        img[self.food_mask] = 2
        img[self.agent_position] = 4
        return img

    def set_areas(self, **areas_generator):
        self.areas = Areas(shape=self.shape, **areas_generator)

    def generate_areas(self):
        self.areas.generate(self.seed)

    @staticmethod
    def make(
            shape_xy: Tuple[int, int], seed: int,
            action_costs: Dict,
            **environment
    ):
        state = EnvironmentState(
            shape_xy=shape_xy, seed=seed
        )
        state.add_action_costs(**action_costs)
        state.set_areas(**environment.get('areas', dict()))
        state.set_obstacles(**environment['obstacles'])

        state.generate_areas()
        state.generate_obstacles()
        state.generate_food(**environment['food'])

        state.set_rendering(**environment['rendering'])
        state.spawn_agent()
        return state

