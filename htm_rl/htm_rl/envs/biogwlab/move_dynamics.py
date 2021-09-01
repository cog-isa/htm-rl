from typing import Tuple

import numpy as np

MOVE_DIRECTIONS = {
    # (i, j)-based coordinates [or (y, x) for the viewer]
    'right': (0, 1), 'down': (1, 0), 'left': (0, -1), 'up': (-1, 0)
}
DIRECTIONS_ORDER = ['right', 'down', 'left', 'up']
TURN_DIRECTIONS = {'right': 1, 'left': -1}


class MoveDynamics:
    @staticmethod
    def turn(view_direction: int, turn_direction: int):
        """Turns view direction."""
        new_direction = (view_direction + turn_direction + 4) % 4
        return new_direction

    @staticmethod
    def try_move(
            position: Tuple[int, int], move_direction: Tuple[int, int],
            shape: Tuple[int, int], obstacle_mask: np.ndarray
    ):
        """
        Performs the move if it's allowed.
        Returns new position and flag whether or not the move was successful.
        """
        new_position = MoveDynamics.move(position, move_direction)
        new_position = MoveDynamics.clip2d(new_position, shape)

        success = MoveDynamics.is_move_successful(position, new_position, obstacle_mask)
        if not success:
            new_position = position
        return new_position, success

    @staticmethod
    def is_move_successful(
            position: Tuple[int, int], new_position: Tuple[int, int],
            obstacle_mask: np.ndarray
    ):
        """Checks whether move is happened and is allowed."""
        # changed position but not stepped into the wall
        return new_position != position and not obstacle_mask[new_position]

    @staticmethod
    def move(position: Tuple[int, int], direction: Tuple[int, int]):
        """
        Calculates new position for the move to the specified direction
        without checking whether this move is allowed or not.
        """
        i, j = position
        i += direction[0]
        j += direction[1]
        return i, j

    @staticmethod
    def clip2d(position: Tuple[int, int], shape: Tuple[int, int]):
        """Clip position to be inside specified rectangle."""
        def clip(x, high):
            if x >= high:
                return high - 1
            if x < 0:
                return 0
            return x
        i, j = position
        i = clip(i, shape[0])
        j = clip(j, shape[1])
        return i, j
