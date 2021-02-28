from typing import Tuple

import numpy as np
from numpy.core._multiarray_umath import ndarray

from htm_rl.common.utils import clip

MOVE_DIRECTIONS = {
    # (i, j)-based coordinates [or (y, x) for the viewer]
    'right': (0, 1), 'down': (1, 0), 'left': (0, -1), 'up': (-1, 0)
}
DIRECTIONS_ORDER = ['right', 'down', 'left', 'up']
TURN_DIRECTIONS = {'right': 1, 'left': -1}


class MoveDynamics:
    @staticmethod
    def turn(view_direction, turn_direction):
        new_direction = (view_direction + turn_direction + 4) % 4
        return new_direction

    @classmethod
    def try_move(cls, position, direction, shape, obstacle_mask):
        new_position = cls.move(position, direction)
        new_position = cls.clip2d(new_position, shape)
        success = cls.is_move_successful(position, new_position, obstacle_mask)
        if not success:
            new_position = position
        return new_position, success

    @staticmethod
    def is_move_successful(position, new_position, obstacle_mask):
        return new_position != position and not obstacle_mask[new_position]

    @staticmethod
    def move(position, direction):
        i, j = position
        i += direction[0]
        j += direction[1]
        return i, j

    @staticmethod
    def clip2d(position, shape):
        i, j = position
        i = clip(i, shape[0])
        j = clip(j, shape[1])
        return i, j