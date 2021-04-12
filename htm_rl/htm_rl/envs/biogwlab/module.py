from enum import Flag, auto

import numpy as np


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
    NonEmpty = Obstacle | Consumable | Agent


class Entity(Module):
    family: str = '???'
    type: EntityType = EntityType.Unknown

    def __init__(self, name: str):
        super(Entity, self).__init__(name=name)

    def append_mask(self, mask: np.ndarray):
        raise NotImplementedError()

    def append_position(self, exist: bool, position):
        raise NotImplementedError()
