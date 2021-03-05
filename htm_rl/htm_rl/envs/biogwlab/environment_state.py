from copy import copy
from typing import Tuple, Optional, Dict, List

import numpy as np

from htm_rl.common.plot_utils import plot_grid_images
from htm_rl.common.sdr_encoders import IntBucketEncoder, SdrConcatenator
from htm_rl.common.utils import isnone
from htm_rl.envs.biogwlab.areas import Areas
from htm_rl.envs.biogwlab.food import Food
from htm_rl.envs.biogwlab.obstacles import Obstacles
from htm_rl.envs.biogwlab.move_dynamics import (
    MOVE_DIRECTIONS, DIRECTIONS_ORDER, TURN_DIRECTIONS, MoveDynamics,
)
from htm_rl.envs.biogwlab.view_clipper import ViewClipper


class EnvironmentState:
    supported_actions = [
        'stay',
        'move left', 'move up', 'move right', 'move down',
        'move forward', 'turn left', 'turn right'
    ]

    seed: int
    shape: Tuple[int, int]
    areas: Areas
    obstacles: Obstacles
    food: Food

    agent_position: Tuple[int, int]
    agent_view_direction: Optional[int]
    episode_step: int
    step_reward: float

    action_cost: float
    action_weight: Dict[str, float]

    output_sdr_size: int

    actions: List[str]
    _episode_max_steps: int

    _render: List[str]
    _encoders: Dict
    _cached_renderings: Dict
    _encoding_sdr_concatenator: SdrConcatenator
    _view_clipper: Optional[ViewClipper]

    def __init__(self, shape_xy: Tuple[int, int], seed: int):
        # convert from x,y to i,j
        width, height = shape_xy
        self.shape = (height, width)

        self.seed = seed
        self.episode_step = 0
        self.step_reward = 0

    def reset(self):
        self.episode_step = 0
        self.step_reward = 0

        self.food.reset()
        self.spawn_agent()

    def observe(self):
        reward = self.step_reward
        obs = self.render()
        is_first = self.episode_step == 0

        plot_grid_images([self.render_rgb()])
        return reward, obs, is_first

    def act(self, action: int):
        action = self.actions[action]

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
        self.step_reward += self.action_weight['turn'] * self.action_cost

    def collect(self):
        if self.food.mask[self.agent_position]:
            self.food.mask[self.agent_position] = False
            if self.food.n_types == 1:
                reward = self.food.rewards[0]
            else:
                reward = self.food.rewards[self.food.map[self.agent_position]]

            if reward > 0:
                self.food.n_foods_to_find -= 1

            self.step_reward += reward

    def is_terminal(self):
        no_foods = self.food.n_foods_to_find <= 0
        steps_exceed = self.episode_step >= self._episode_max_steps
        return steps_exceed or no_foods

    def set_obstacles(self, **generator):
        self.obstacles = Obstacles(shape=self.shape, seed=self.seed, **generator)

    def generate_obstacles(self):
        self.obstacles.generate()

    def _flatten_position(self, position):
        i, j = position
        return i * self.shape[1] + j

    def _unflatten_position(self, flatten_position):
        return divmod(flatten_position, self.shape[1])

    def set_food(self, **food):
        self.food = Food(shape=self.shape, **food)

    def generate_food(self):
        self.food.generate(self.seed, self.obstacles.mask, self.areas.map)

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

    def set_action_costs(self, action_cost: float, action_weight: Dict[str, float]):
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
            encoder = None
            if data_name == 'position':
                n_cells = self.shape[0] * self.shape[1]
                encoder = IntBucketEncoder(n_values=n_cells, **renderer)
            elif data_name == 'direction':
                encoder = IntBucketEncoder(n_values=len(MOVE_DIRECTIONS), **renderer)
            elif data_name == 'obstacles':
                encoder = self.obstacles.set_renderer(view_shape)
            elif data_name == 'food':
                encoder = self.food.set_renderer(view_shape)
            elif data_name == 'area':
                encoder = self.areas.set_renderer(view_shape)

            if encoder is not None:
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
                encoded_data = self.food.render(view_clip)
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
        self.areas.render_rgb(img)
        self.obstacles.render_rgb(img)
        self.food.render_rgb(img)

        img[self.agent_position] = 24
        return img

    def set_areas(self, **areas_generator):
        self.areas = Areas(shape=self.shape, **areas_generator)

    def generate_areas(self):
        self.areas.generate(self.seed)

    @classmethod
    def ensure_all_actions_supported(cls, actions):
        non_supported_actions = [
            action for action in actions
            if action not in EnvironmentState.supported_actions
        ]
        assert not non_supported_actions, \
            f'{non_supported_actions} actions are not supported'

    def set_actions(self, actions):
        self.actions = isnone(actions, self.supported_actions.copy())
        self.ensure_all_actions_supported(self.actions)

    def set_regenerator(self, episode_max_steps: int = None):
        h, w = self.shape
        self._episode_max_steps = isnone(episode_max_steps, 2 * h * w)
