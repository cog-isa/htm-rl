from abc import ABC, abstractmethod
from itertools import product
from typing import Tuple, List

import matplotlib.pyplot as plt
import numpy as np

from htm_rl.common.utils import timed, trace, clip
from htm_rl.envs.biogwlab.environment_state import BioGwLabEnvState, EnvironmentState


class BaseIncrementalGenerator(ABC):
    @abstractmethod
    def generate(self, seed: int, env_state: EnvironmentState):
        ...


class EnvironmentGenerator:
    shape: Tuple[int, int]
    seed: int

    def __init__(self, shape: Tuple[int, int], seed: int):
        self.shape = shape
        self.seed = seed

    def generate(self):
        environment_state = EnvironmentState(self.shape, self.seed)
        return environment_state

