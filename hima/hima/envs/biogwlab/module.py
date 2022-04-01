from enum import Flag, auto
from typing import Tuple

import numpy as np

from hima.envs.biogwlab.view_clipper import ViewClip


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

    rendering: bool
    initialized: bool

    def __init__(self, name: str, rendering: bool = True):
        super(Entity, self).__init__(name=name)
        self.rendering = rendering
        self.initialized = False

    def render(self, view_clip: ViewClip = None):
        raise NotImplementedError()

    def append_mask(self, mask: np.ndarray):
        raise NotImplementedError()

    def append_position(self, exist: bool, position):
        raise NotImplementedError()
