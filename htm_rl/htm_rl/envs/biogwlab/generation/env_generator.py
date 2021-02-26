from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np

from htm_rl.envs.biogwlab.environment_state import EnvironmentState
from htm_rl.envs.biogwlab.generation.food import FoodGenerator
from htm_rl.envs.biogwlab.generation.obstacles import ObstacleGenerator


class BaseIncrementalGenerator(ABC):
    @abstractmethod
    def generate(self, seed: int, env_state: EnvironmentState):
        ...


class EnvironmentGenerator:
    _shape_xy: Tuple[int, int]
    _obstacle_density: float

    def __init__(self, shape_xy: Tuple[int, int], seed: int, obstacle_density: float):
        self._shape_xy = shape_xy
        self._obstacle_density = obstacle_density
        self.seed_generator = np.random.default_rng(seed)

    def generate(self):
        seed = self.seed_generator.integers(100000)
        state = EnvironmentState(self._shape_xy, seed)
        ObstacleGenerator(state.shape, self._obstacle_density).add(state)
        FoodGenerator().add(state)
        AgentPositionGenerator().add(state)
        return state


class AgentPositionGenerator:
    def add(self, state: EnvironmentState):
        rnd = np.random.default_rng(state.seed)

        empty_mask = ~(state.obstacle_mask | state.food_mask)
        available_positions_fl = np.flatnonzero(empty_mask)
        position_fl = rnd.choice(available_positions_fl)
        position = divmod(position_fl, state.shape[1])

        state.agent_position = position
