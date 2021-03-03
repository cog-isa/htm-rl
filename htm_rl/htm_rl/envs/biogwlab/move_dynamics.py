from typing import Tuple

import numpy as np

from htm_rl.common.utils import clip

MOVE_DIRECTIONS = {
    # (i, j)-based coordinates [or (y, x) for the viewer]
    'right': (0, 1), 'down': (1, 0), 'left': (0, -1), 'up': (-1, 0)
}
DIRECTIONS_ORDER = ['right', 'down', 'left', 'up']
TURN_DIRECTIONS = {'right': 1, 'left': -1}


class MoveDynamics:
    @staticmethod
    def turn(view_direction: int, turn_direction: int):
        new_direction = (view_direction + turn_direction + 4) % 4
        return new_direction

    @classmethod
    def try_move(
            cls, position: Tuple[int, int], move_direction: Tuple[int, int],
            shape: Tuple[int, int], obstacle_mask: np.ndarray
    ):
        new_position = cls.move(position, move_direction)
        new_position = cls.clip2d(new_position, shape)
        success = cls.is_move_successful(position, new_position, obstacle_mask)
        if not success:
            new_position = position
        return new_position, success

    @staticmethod
    def is_move_successful(
            position: Tuple[int, int], new_position: Tuple[int, int],
            obstacle_mask: np.ndarray
    ):
        # changed position but not into the wall
        return new_position != position and not obstacle_mask[new_position]

    @staticmethod
    def move(position: Tuple[int, int], direction: Tuple[int, int]):
        i, j = position
        i += direction[0]
        j += direction[1]
        return i, j

    @staticmethod
    def clip2d(position: Tuple[int, int], shape: Tuple[int, int]):
        i, j = position
        i = clip(i, shape[0])
        j = clip(j, shape[1])
        return i, j