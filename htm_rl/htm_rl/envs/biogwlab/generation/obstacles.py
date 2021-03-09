from typing import Tuple

import numpy as np

from htm_rl.envs.biogwlab.move_dynamics import MOVE_DIRECTIONS, DIRECTIONS_ORDER, MoveDynamics


class ObstaclesGenerator:
    density: float
    shape: Tuple[int, int]

    def __init__(self, density: float, shape):
        self.density = density
        self.shape = shape

    def generate(self, seed):
        height, width = self.shape
        n_cells = height * width
        n_required_obstacles = int((1. - self.density) * n_cells)

        rng = np.random.default_rng(seed=seed)

        obstacle_mask = np.zeros(self.shape, dtype=np.bool)
        non_visited_neighbors = np.empty_like(obstacle_mask, dtype=np.float)

        p_change_cell = n_cells ** -.25
        p_move_forward = 1. - n_cells ** -.375

        position = self._centered_rand2d(height, width, rng)
        view_direction = rng.choice(4)
        obstacle_mask[position] = True
        n_obstacles = 1

        while n_obstacles < n_required_obstacles:
            success = False
            if rng.random() < p_move_forward:
                direction = MOVE_DIRECTIONS[DIRECTIONS_ORDER[view_direction]]
                position, success = MoveDynamics.try_move(position, direction, self.shape, obstacle_mask)
                if success:
                    obstacle_mask[position] = True
                    n_obstacles += 1

            if not success:
                turn_direction = self._get_random_turn(rng)
                view_direction = MoveDynamics.turn(view_direction, turn_direction)

            if rng.random() < p_change_cell:
                position, view_direction = self._choose_rnd_cell(obstacle_mask, non_visited_neighbors, rng)
        return ~obstacle_mask

    def _centered_rand2d(self, high_i, high_j, rng):
        def centered_random(high):
            return int(high/4 + rng.random() * high/2)

        i = centered_random(high_i)
        j = centered_random(high_j)
        return i, j

    def _get_random_turn(self, rng):
        return int(np.sign(.5 - rng.random()))

    def _choose_rnd_cell(self, gridworld: np.ndarray, non_visited_neighbors: np.ndarray, rng):
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
        cell_flatten_index = rng.choice(flatten_visited_indices, p=probabilities)
        i, j = divmod(cell_flatten_index, self.shape[1])

        # choose direction
        view_direction = rng.choice(4)
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