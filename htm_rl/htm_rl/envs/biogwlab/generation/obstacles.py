from typing import Tuple

import numpy as np
from numpy.random._generator import Generator

from htm_rl.common.sdr_encoders import IntArrayEncoder
from htm_rl.envs.biogwlab.move_dynamics import MOVE_DIRECTIONS, DIRECTIONS_ORDER, MoveDynamics


class ObstacleGenerator:
    density: float
    shape: Tuple[int, int]
    rng: Generator

    def __init__(self, density: float, shape, seed):
        self.density = density
        self.shape = shape
        self.rng = np.random.default_rng(seed=seed)

    def generate(self):
        height, width = self.shape
        n_cells = height * width
        n_required_obstacles = int((1. - self.density) * n_cells)

        obstacle_mask = np.zeros(self.shape, dtype=np.bool)
        non_visited_neighbors = np.empty_like(obstacle_mask, dtype=np.float)

        p_change_cell = n_cells ** -.25
        p_move_forward = 1. - n_cells ** -.375

        position = self._centered_rand2d(height, width)
        view_direction = self.rng.choice(4)
        obstacle_mask[position] = True
        n_obstacles = 1

        while n_obstacles < n_required_obstacles:
            success = False
            if self.rng.random() < p_move_forward:
                direction = MOVE_DIRECTIONS[DIRECTIONS_ORDER[view_direction]]
                position, success = MoveDynamics.try_move(position, direction, self.shape, obstacle_mask)
                if success:
                    obstacle_mask[position] = True
                    n_obstacles += 1

            if not success:
                turn_direction = self._get_random_turn()
                view_direction = MoveDynamics.turn(view_direction, turn_direction)

            if self.rng.random() < p_change_cell:
                position, view_direction = self._choose_rnd_cell(obstacle_mask, non_visited_neighbors)
        return ~obstacle_mask

    def _centered_rand2d(self, high_i, high_j):
        def centered_random(high):
            return int(high/4 + self.rng.random() * high/2)

        i = centered_random(high_i)
        j = centered_random(high_j)
        return i, j

    def _get_random_turn(self):
        return int(np.sign(.5 - self.rng.random()))

    def _choose_rnd_cell(self, gridworld: np.ndarray, non_visited_neighbors: np.ndarray):
        # count non-visited neighbors
        non_visited_neighbors.fill(0)
        non_visited_neighbors[1:] += gridworld[1:] * (~gridworld[:-1])
        non_visited_neighbors[:-1] += gridworld[:-1] * (~gridworld[1:])
        non_visited_neighbors[:, 1:] += gridworld[:, 1:] * (~gridworld[:, :-1])
        non_visited_neighbors[:, :-1] += gridworld[:, :-1] * (~gridworld[:, 1:])
        # normalize to become probabilities
        non_visited_neighbors /= non_visited_neighbors.sum()

        # choose cell
        flatten_visited_indices = np.flatnonzero(non_visited_neighbors)
        probabilities = non_visited_neighbors.ravel()[flatten_visited_indices]
        cell_flatten_index = self.rng.choice(flatten_visited_indices, p=probabilities)
        i, j = divmod(cell_flatten_index, self.shape[1])

        # choose direction
        view_direction = self.rng.choice(4)
        return (i, j), view_direction


class WallColorGenerator:
    COLOR_DISTRIBUTION = [
        [.8, .2, .0, .0],
        [.3, .5, .1, .1],
        [.0, .1, .7, .2],
        [.1, .0, .4, .5],
        [.25, .15, .2, .4]
    ]

    def __init__(self):
        self.color_distr = np.array(self.COLOR_DISTRIBUTION, dtype=np.float)

        assert np.abs(self.color_distr.sum(axis=1) - 1.).sum() < 1e-5, 'Check each row sums to 1!'
        self.n_colors = self.color_distr.shape[0]

    def color_walls(self, obstacle_mask, areas_map, seed):
        rnd = np.random.default_rng(seed=seed)

        wall_colors = np.full_like(areas_map, -1, dtype=np.int8)
        for area in range(areas_map.max() + 1):
            mask = (areas_map == area) & obstacle_mask
            wall_colors[mask] = rnd.choice(
                self.color_distr.shape[1],
                p=self.color_distr[area % self.n_colors],
                size=mask.sum()
            )
        # trace_image(self.verbosity, 2, wall_colors + 1)
        return wall_colors, self.n_colors


class Obstacles:
    shape: Tuple[int, int]
    view_shape: Tuple[int, int]

    mask: np.ndarray
    map: np.ndarray

    _generator: ObstacleGenerator
    _encoder: IntArrayEncoder

    def __init__(self, shape, seed, **generator):
        self.shape = shape
        self._generator = ObstacleGenerator(shape=self.shape, seed=seed, **generator)

    def generate(self):
        self.mask = self._generator.generate()
        self.map = (~self.mask).astype(np.int)

    def set_renderer(self, view_shape):
        self.view_shape = view_shape
        self._encoder = IntArrayEncoder(shape=view_shape, n_types=1)
        return self._encoder

    def render(self, view_clip=None):
        if view_clip is not None:
            view_indices, abs_indices = view_clip

            view_mask = np.ones(self.view_shape, dtype=np.bool).flatten()
            view_mask[view_indices] = self.mask.flatten()[abs_indices]

            view_map = np.zeros(self.view_shape, dtype=np.int).flatten()
            view_map[view_indices] = self.map.flatten()[abs_indices]
            return self._encoder.encode(view_map, view_mask)
        else:
            return self._encoder.encode(self.map)
