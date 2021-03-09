from typing import Tuple, Dict, List, Any

import numpy as np

from htm_rl.common.utils import isnone
from htm_rl.envs.biogwlab.agent import Agent
from htm_rl.envs.biogwlab.move_dynamics import (
    MOVE_DIRECTIONS, DIRECTIONS_ORDER, TURN_DIRECTIONS,
)


class EnvironmentState:
    supported_actions = [
        'stay',
        'move left', 'move up', 'move right', 'move down',
        'move forward', 'turn left', 'turn right'
    ]
    supported_entities = {'obstacles', 'areas', 'food', 'agent'}
    supported_events = [
        'generate',
        'move', 'turn', 'collect',
        'render'
    ]

    seed: int
    shape: Tuple[int, int]

    modules: Dict[str, Any]
    handlers: Dict[str, List[Any]]

    agent: Agent
    episode_step: int
    step_reward: float

    action_cost: float
    action_weight: Dict[str, float]

    actions: List[str]
    _episode_max_steps: int

    def __init__(self, shape_xy: Tuple[int, int], seed: int):
        # convert from x,y to i,j
        width, height = shape_xy
        self.shape = (height, width)

        self.modules = dict()
        self.handlers = dict()

        self.seed = seed
        self.episode_step = 0
        self.step_reward = 0

    def reset(self):
        self.episode_step = 0
        self.step_reward = 0
        self.generate()

    def add_module(self, name, module):
        self.modules[name] = module

        for event in self.supported_events:
            if hasattr(module, event):
                handlers = self.handlers.setdefault(event, [])
                handlers.append(getattr(module, event))

    def get_module(self, name):
        return self.modules[name]

    def add_agent(self):
        obstacles = self.get_module('obstacles')
        self.agent = Agent(env=self, obstacles=obstacles)
        self.add_module('agent', self.agent)

    def generate(self):
        rng = np.random.default_rng(self.seed)
        for handler in self.handlers['generate']:
            seed = rng.integers(100000)
            handler(seed)

    def observe(self):
        reward = self.step_reward
        obs = self.render()
        is_first = self.episode_step == 0

        # plot_grid_images([self.render_rgb()])
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
                direction = DIRECTIONS_ORDER[self.agent.view_direction]

            direction = MOVE_DIRECTIONS[direction]
            self.move(direction)

        self.collect()
        self.episode_step += 1

    def stay(self):
        self.step_reward += self.action_weight['stay'] * self.action_cost

    def move(self, direction):
        for handler in self.handlers['move']:
            handler(direction)

        self.step_reward += self.action_weight['move'] * self.action_cost

    def turn(self, turn_direction):
        for handler in self.handlers['turn']:
            handler(turn_direction)

        self.step_reward += self.action_weight['turn'] * self.action_cost

    def collect(self):
        for handler in self.handlers['collect']:
            reward = handler(self.agent.position, self.agent.view_direction)
            self.step_reward += reward

    def is_terminal(self):
        steps_exceed = self.episode_step >= self._episode_max_steps
        return steps_exceed

    def set_action_costs(self, action_cost: float, action_weight: Dict[str, float]):
        self.action_cost = action_cost
        self.action_weight = action_weight

    def render(self):
        for handler in self.handlers['render']:
            return handler(self.agent.position, self.agent.view_direction)

    @property
    def output_sdr_size(self):
        renderer = self.get_module('rendering')
        return renderer.output_sdr_size

    def render_rgb(self):
        img = np.zeros(self.shape, dtype=np.int8)

        areas = self.get_module('areas')
        img[:] = areas.map

        obstacles = self.get_module('obstacles')
        img[obstacles.mask] = 8

        food = self.get_module('food')
        norm_rewards = 5 * food.rewards[food.map[food.mask]] / np.abs(food.rewards).max()
        norm_rewards[norm_rewards > 0] = norm_rewards[norm_rewards > 0].astype(np.int8) + 12
        norm_rewards[norm_rewards < 0] = norm_rewards[norm_rewards < 0].astype(np.int8) - 4
        img[food.mask] = norm_rewards

        img[self.agent.position] = 24
        return img

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
