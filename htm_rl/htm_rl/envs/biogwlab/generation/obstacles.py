from typing import Tuple

import numpy as np

from htm_rl.common.utils import clip
from htm_rl.envs.biogwlab.environment_state import EnvironmentState


class ObstacleGenerator:
    # in x, y coords
    directions = [(1, 0), (0, -1), (-1, 0), (0, 1)]

    # in i, j coords
    shape: Tuple[int, int]
    density: float

    def __init__(self, shape: Tuple[int, int], density: float):
        self.shape = shape
        self.density = density

    def add(self, state: EnvironmentState):
        height, width = state.shape
        rnd = np.random.default_rng(seed=state.seed)

        n_cells = state.n_cells
        n_required_obstacles = int((1. - self.density) * n_cells)

        clear_path_mask = np.zeros(self.shape, dtype=np.bool)
        non_visited_neighbors = np.empty_like(clear_path_mask, dtype=np.float)

        p_change_cell = n_cells ** -.25
        p_move_forward = 1. - n_cells ** -.375

        i, j = self._centered_rand2d(height, width, rnd)
        view_direction = rnd.integers(4)
        clear_path_mask[i][j] = True
        n_obstacles = 0

        while n_obstacles < n_required_obstacles:
            moved_forward = False
            if rnd.random() < p_move_forward:
                _i, _j = self._try_move_forward(i, j, view_direction)
                if not clear_path_mask[_i][_j]:
                    i, j = _i, _j
                    clear_path_mask[i, j] = True
                    n_obstacles += 1
                    moved_forward = True

            if not moved_forward:
                view_direction = self._turn(view_direction, rnd)

            if rnd.random() < p_change_cell:
                i, j, cell_flatten_index = self._choose_rnd_cell(
                    clear_path_mask, non_visited_neighbors, rnd
                )

        obstacle_mask = ~clear_path_mask
        state.obstacle_mask = obstacle_mask

    def _try_move_forward(self, i, j, view_direction):
        j += self.directions[view_direction][0]     # X axis
        i += self.directions[view_direction][1]     # Y axis

        i = clip(i, self.shape[0])
        j = clip(j, self.shape[1])
        return i, j

    @staticmethod
    def _turn(view_direction, rnd):
        turn_direction = int(np.sign(.5 - rnd.random()))
        return (view_direction + turn_direction + 4) % 4

    def _choose_rnd_cell(
            self, gridworld: np.ndarray, non_visited_neighbors: np.ndarray,
            rnd: np.random.Generator
    ):
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
        cell_flatten_index = rnd.choice(flatten_visited_indices, p=probabilities)
        i, j = divmod(cell_flatten_index, self.shape[1])

        # choose direction
        view_direction = rnd.integers(4)
        return i, j, view_direction

    @staticmethod
    def _centered_rand2d(max_i, max_j, rnd):
        mid_i = (max_i + 1)//2
        mid_j = (max_j + 1)//2

        i = mid_i + rnd.integers(-max_i//4, max_i//4 + 1)
        j = mid_j + rnd.integers(-max_j//4, max_j//4 + 1)
        return i, j


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
