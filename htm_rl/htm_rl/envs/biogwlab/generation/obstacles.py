import numpy as np

from htm_rl.common.utils import timed, trace, clip



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



class ObstacleGenerator:
    directions = [(1, 0), (0, -1), (-1, 0), (0, 1)]

    size: int
    density: float
    verbosity: int

    def __init__(self, size: int, density: float, verbosity: int):
        self.size = size
        self.density = density
        self.verbosity = verbosity

    @timed
    def generate(self, seed):
        n = self.size
        required_cells = int((1. - self.density) * n**2)
        rnd = np.random.default_rng(seed=seed)

        clear_path_mask = np.zeros((n, n), dtype=np.bool)
        non_visited_neighbors = np.empty_like(clear_path_mask, dtype=np.float)

        p_change_cell = .1/np.sqrt(n)
        p_move_forward = 1. - 1./(n ** 0.75)

        i, j = self._rand2d(n//2, (n+1)//2, rnd)
        view_direction = rnd.integers(4)
        clear_path_mask[i][j] = True
        n_cells, n_iterations = 0, 0

        while n_cells < required_cells:
            n_iterations += 1
            moved_forward = False
            if rnd.random() < p_move_forward:
                _i, _j = self._try_move_forward(i, j, view_direction)
                if not clear_path_mask[_i][_j]:
                    i, j = _i, _j
                    clear_path_mask[i, j] = True
                    n_cells += 1
                    moved_forward = True

            if not moved_forward:
                view_direction = self._turn(view_direction, rnd)

            if rnd.random() < p_change_cell:
                i, j, cell_flatten_index = self._choose_rnd_cell(clear_path_mask, non_visited_neighbors, rnd)

        obstacle_mask = ~clear_path_mask
        trace(self.verbosity, 3, f'Gridworld map generation efficiency: {n_iterations / (n**2)}')
        trace_image(self.verbosity, 3, obstacle_mask)

        return obstacle_mask

    def _try_move_forward(self, i, j, view_direction):
        j += self.directions[view_direction][0]     # X axis
        i += self.directions[view_direction][1]     # Y axis

        i = clip(i, self.size)
        j = clip(j, self.size)
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

    def _rand2d(self, size, shift, rnd):
        i, j = divmod(rnd.integers(size**2), size)
        i, j = i + shift, j + shift
        return i, j
