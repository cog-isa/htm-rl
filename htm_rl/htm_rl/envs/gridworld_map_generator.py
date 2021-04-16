from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from htm_rl.common.utils import timed, trace
from htm_rl.envs.gridworld_mdp import GridworldMdp
from htm_rl.envs.gridworld_pomdp import GridworldPomdp


class GridworldMapGenerator:
    directions = [(1, 0), (0, -1), (-1, 0), (0, 1)]

    seed: int
    size: int
    density: float
    verbosity: int
    view_radius: Optional[int]
    current_env: Optional[GridworldMdp]

    def __init__(
            self, seed: int, size: int, density: float, verbosity: int,
            view_radius: int = None
    ):
        self.seed = seed
        self.size = size
        self.density = density
        self.verbosity = verbosity
        self.view_radius = view_radius
        self.current_env = None
        self.current_mode = 'mdp' if self.view_radius is None else 'pomdp'
        trace(self.verbosity, 1, f'Gridworld mode: {self.current_mode}')

    def __iter__(self):
        mode = self.current_mode
        rnd_generator = np.random.default_rng(seed=self.seed)
        while True:
            seed = rnd_generator.integers(2**31)
            gridworld_map, t = self.generate(seed)
            trace(self.verbosity, 3, f'Gridworld {mode} {seed} generated in {t:.5f} sec')
            if mode == 'mdp':
                self.current_env = GridworldMdp(gridworld_map, seed)
            else:
                self.current_env = GridworldPomdp(self.view_radius, gridworld_map, seed)
            yield self.current_env

    @timed
    def generate(self, seed):
        n = self.size
        required_cells = int(self.density * n**2)
        rnd = np.random.default_rng(seed=seed)

        gridworld = np.zeros((n, n), dtype=np.bool)
        non_visited_neighbors = np.empty_like(gridworld, dtype=np.float)

        p_change_cell = .1/np.sqrt(n)
        p_move_forward = 1. - 1./(n ** 0.75)

        i, j = divmod(rnd.integers(n**2), n)
        view_direction = rnd.integers(4)
        gridworld[i][j] = True
        n_cells, n_iterations = 0, 0

        while n_cells < required_cells:
            n_iterations += 1
            moved_forward = False
            if rnd.random() < p_move_forward:
                _i, _j = self._try_move_forward(i, j, view_direction)
                if not gridworld[_i][_j]:
                    i, j = _i, _j
                    gridworld[i, j] = True
                    n_cells += 1
                    moved_forward = True

            if not moved_forward:
                view_direction = self._turn(view_direction, rnd)

            if rnd.random() < p_change_cell:
                i, j, cell_flatten_index = self._choose_rnd_cell(
                    gridworld, non_visited_neighbors, rnd
                )

        trace(self.verbosity, 3, f'Gridworld map generation efficiency: {n_iterations / (n**2)}')
        if self.verbosity >= 4:
            plt.imshow(gridworld)
            plt.show(block=False)

        return gridworld

    def _try_move_forward(self, i, j, view_direction):
        i += self.directions[view_direction][0]
        j += self.directions[view_direction][1]

        i, j = max(0, min(self.size - 1, i)), max(0, min(self.size - 1, j))
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
        i, j = divmod(cell_flatten_index, self.size)

        # choose direction
        view_direction = rnd.integers(4)
        return i, j, view_direction
