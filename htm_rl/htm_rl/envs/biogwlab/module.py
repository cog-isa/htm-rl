from enum import Enum, Flag, auto
from typing import Tuple, Optional

import numpy as np

from htm_rl.common.utils import isnone
from htm_rl.envs.biogwlab.environment import Environment


class Module:
    """Basic module that extends Environment functionality."""
    name: str

    def __init__(self, name: str = '???'):
        self.name = name


class EntityType(Flag):
    Unknown = 0
    Area = auto()
    Obstacle = auto()
    Consumable = auto()
    Agent = auto()


class Entity(Module):
    family: str = '???'
    type: EntityType = EntityType.Unknown

    def __init__(self, name: str):
        super(Entity, self).__init__(name=name)

    def append_mask(self, mask: np.ndarray):
        raise NotImplementedError()

    def append_position(self, exist: bool, position):
        raise NotImplementedError()
