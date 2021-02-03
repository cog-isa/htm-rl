from abc import ABC, abstractmethod
from itertools import product
from typing import Tuple, List

import matplotlib.pyplot as plt
import numpy as np

from htm_rl.common.utils import timed, trace, clip
from htm_rl.envs.biogwlab.env_state import BioGwLabEnvState, EnvironmentState


class BaseIncrementalGenerator(ABC):
    @abstractmethod
    def generate(self, seed: int, env_state: EnvironmentState):
        ...


class EnvironmentGenerator:
    size: Tuple[int, int]
    verbosity: int
    seed: int
    generation_pipeline: List[BaseIncrementalGenerator]

    def __init__(
            self, size: Tuple[int, int], verbosity: int,
            seed: int, generation_pipeline: List[BaseIncrementalGenerator]
    ):
        self.size = size
        self.verbosity = verbosity

        self.seed = seed
        self.generation_pipeline = generation_pipeline

    def generate(
            self, seed=None,
            generation_pipeline: List[BaseIncrementalGenerator] = None
    ):
        seed = self.seed
        generation_pipeline = self.generation_pipeline

        environment_state = EnvironmentState(self.size, seed)
        for incremental_generator in generation_pipeline:
            incremental_generator.generate(seed, environment_state)
        return environment_state

