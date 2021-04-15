from typing import List, Dict

from htm_rl.envs.biogwlab.environment import Environment
from htm_rl.envs.biogwlab.module import Module


class ActionsCost(Module):
    base_coast: float
    weights: Dict[str, float]

    env: Environment

    def __init__(
            self, env: Environment, base_cost: float, weights: Dict[str, float],
            **module
    ):
        super().__init__(**module)

        self.base_coast = base_cost
        self.weights = weights
        self.env = env

    def stay(self):
        self.env.step_reward += self.weights['stay'] * self.base_coast

    def move(self, direction):
        self.env.step_reward += self.weights['move'] * self.base_coast

    def turn(self, turn_direction: int):
        self.env.step_reward += self.weights['turn'] * self.base_coast
