from typing import Any

import numpy as np

from htm_rl.common.utils import isnone
from htm_rl.envs.biogwlab.agent import Agent
from htm_rl.envs.biogwlab.flags import EntityDict, EntityMaskAggregation, TEntityDict, TEntityKey
from htm_rl.envs.biogwlab.module import Module, Entity, EntityType
from htm_rl.envs.biogwlab.move_dynamics import (
    MOVE_DIRECTIONS, DIRECTIONS_ORDER, TURN_DIRECTIONS,
)
from htm_rl.envs.biogwlab.renderer import Renderer
from htm_rl.envs.env import Env


class Environment(Env):
    """
    Core implementation of the BioGridworldLab environment. It requires manual building, so
    you may consider to create a wrapper that implement building process and use it instead.
    """

    supported_actions = [
        'stay',
        'move left', 'move up', 'move right', 'move down',
        'move forward', 'turn left', 'turn right'
    ]
    supported_entities = {'obstacle', 'area', 'food', 'agent'}
    supported_events = [
        'reset', 'generate', 'generate_seeds',
        'move', 'turn', 'collect',
        'collected',
        'is_terminal',
    ]

    seed: int
    shape: tuple[int, int]

    agent: Agent
    modules: dict[str, Module]
    handlers: dict[str, list[Any]]

    entities: TEntityDict
    aggregated_mask: dict[TEntityKey, np.ndarray]

    episode_step: int
    step_reward: float

    renderer: Renderer

    actions: list[str]

    def __init__(
            self, shape_xy: tuple[int, int], seed: int, actions: list[str],
            rendering: dict = None
    ):
        self.seed = seed
        self.renderer = Renderer(shape_xy=shape_xy, **isnone(rendering, {}))
        self.shape = self.renderer.shape.full_shape

        self.actions = isnone(actions, self.supported_actions.copy())
        ensure_all_actions_supported(self.actions, self.supported_actions)

        self.modules = {}
        self.handlers = {}
        self.entities = EntityDict()
        self.aggregated_mask = EntityMaskAggregation(self.entities, self.shape)

        self.episode_step = 0
        self.step_reward = 0
        self.items_collected = 0

    def reset(self):
        self.episode_step = 0
        self.step_reward = 0
        self.items_collected = 0

        self._run_handlers('reset')
        self.generate()

        # noinspection PyTypeChecker
        self.agent = self.entities['agent']

        self.observe()

    def add_module(self, module: Module):
        self.remove_module(module)

        self.modules[module.name] = module
        if isinstance(module, Entity):
            self.entities[module.name] = module

        for event in self.supported_events:
            if hasattr(module, event):
                handlers = self.handlers.setdefault(event, [])
                handlers.append(getattr(module, event))

    def remove_module(self, module: Module):
        if module.name not in self.modules:
            return

        self.modules.pop(module.name)
        if isinstance(module, Entity):
            self.entities.pop(module.name)

        for event in self.supported_events:
            if hasattr(module, event):
                self.handlers[event].remove(getattr(module, event))

    def get_module(self, name):
        return self.modules[name]

    def _get_from_single_handler(self, key: str, **handler_kwargs):
        for handler in self.handlers.get(key, []):
            return handler(**handler_kwargs)
        return None

    def _run_handlers(self, key: str, **handler_kwargs):
        for handler in self.handlers.get(key, []):
            handler(**handler_kwargs)

    def generate(self):
        seeds = self._get_from_single_handler('generate_seeds')
        self._run_handlers('generate', seeds=seeds)

    def observe(self):
        reward = self.step_reward
        obs = self.render()
        is_first = self.episode_step == 0
        return reward, obs, is_first

    def act(self, action: int):
        if self.is_terminal():
            self.reset()
            return

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
                agent: Agent = self.agent
                direction = DIRECTIONS_ORDER[agent.view_direction]

            direction = MOVE_DIRECTIONS[direction]
            self.move(direction)

        self.collect()
        self.episode_step += 1

    def stay(self):
        self._run_handlers('stay')

    def move(self, direction: int):
        self._run_handlers('move', direction=direction)

    def turn(self, turn_direction: int):
        self._run_handlers('turn', turn_direction=turn_direction)

    def collect(self):
        agent = self.agent
        for handler in self.handlers['collect']:
            reward, success = handler(agent.position, agent.view_direction)
            if success:
                self.step_reward += reward
                self.items_collected += 1
                self.collected()

    def collected(self):
        self._run_handlers('collected')

    def is_terminal(self):
        for handler in self.handlers['is_terminal']:
            if handler(self.episode_step):
                return True
        return False

    def render(self):
        agent = self.agent

        return self.renderer.render(
            position=agent.position, view_direction=agent.view_direction,
            entities=self.entities.values()
        )

    @property
    def n_actions(self):
        return len(self.actions)

    @property
    def output_sdr_size(self):
        return self.renderer.output_sdr_size

    def render_rgb(self, show_outer_walls: bool = False):
        agent = self.agent
        return self.renderer.render_rgb(
            position=agent.position, view_direction=agent.view_direction,
            entities=self.entities,
            show_outer_walls=show_outer_walls
        )


def ensure_all_actions_supported(actions, supported_actions):
    non_supported_actions = [
        action for action in actions
        if action not in supported_actions
    ]
    assert not non_supported_actions, \
        f'{non_supported_actions} actions are not supported'
