from itertools import product

import matplotlib.pyplot as plt
import numpy as np
from numpy.random._generator import Generator

from htm_rl.common.utils import timed, trace, clip


class BioGwLabEnvGenerator:
    size: int
    density: float
    n_areas: int
    seed: int
    verbosity: int

    _rng_seed_generator: Generator

    def __init__(self, size, density, n_areas, seed, verbosity):
        self.size = size
        self.density = density
        self.n_areas = n_areas
        self.seed = seed
        self.verbosity = verbosity

        self._rng_seed_generator = np.random.default_rng(seed)

    def generate(self):
        size = self.size
        seed = self._rng_seed_generator.integers(1_000_000)
        density = self.density
        n_areas = self.n_areas
        obstacle_generator = ObstacleGenerator(size, density, self.verbosity)
        obstacle_mask, _ = obstacle_generator.generate(seed)

        areas_generator = AreasGenerator(size, n_areas, self.verbosity)
        areas_map, n_area_types = areas_generator.generate(seed)

        wall_colorer = WallColorGenerator()
        wall_colors, n_wall_colors = wall_colorer.color_walls(obstacle_mask, areas_map, seed)

        food_generator = FoodGenerator(self.verbosity)
        food_items, food_map, food_mask, n_food_types = food_generator.generate(areas_map, obstacle_mask, seed)
        self._plot_food_map(food_map, obstacle_mask)

        scent_generator = ScentGenerator(self.verbosity)
        food_scents, n_scent_channels = scent_generator.generate(food_items, obstacle_mask)
        self._plot_food_scent_map(food_scents, 0)

        self._plot_final_map(areas_map, obstacle_mask, wall_colors, food_map)
        return BioGwLabEnvState(
            size, seed, n_area_types, n_wall_colors, n_food_types,
            n_scent_channels, obstacle_mask, areas_map, wall_colors,
            food_items, food_map, food_mask, food_scents
        )

    def _plot_final_map(self, areas_map, obstacle_mask, wall_colors, food_map):
        req_verbosity = 3
        if self.verbosity < req_verbosity:
            return
        final_map = areas_map + wall_colors.max(initial=0)*2 + 4
        m = final_map.max()
        final_map[food_map == 0] = m + 5
        final_map[food_map == 2] = m + 10
        final_map[food_map == 1] = -5
        final_map[food_map == 3] = -10
        final_map[obstacle_mask] = wall_colors[obstacle_mask]
        trace_image(self.verbosity, req_verbosity, final_map)

    def _plot_food_scent_map(self, food_scents, channel):
        req_verbosity = 3
        if self.verbosity < req_verbosity:
            return
        food_scent = food_scents.sum(axis=-1)
        food_scent /= (
            food_scent
                .reshape((-1, food_scent._shape_xy[-1]))
                .sum(axis=0)
                .reshape((1, 1, -1))
        )

        trace_image(self.verbosity, req_verbosity, food_scent[:, :, channel])

    def _plot_food_map(self, food_map, obstacle_mask):
        req_verbosity = 3
        if self.verbosity < req_verbosity:
            return
        food_map = food_map.copy()
        food_map += 2
        food_map[obstacle_mask] = 0
        trace_image(self.verbosity, req_verbosity, food_map)


class ScentGenerator:
    BEANS_SCENT_WEIGHT =            [.9, .4, .1, .0, .5, .3]
    RANCID_BEANS_SCENT_WEIGHT =     [.7, .0, .8, .2, .3, .1]
    FRUIT_SCENT_WEIGHT =            [.1, .9, .0, .1, .2, .7]
    RANCID_FRUIT_SCENT_WEIGHT =     [.0, .3, .6, .8, .1, .4]

    scent_weights: np.ndarray
    n_scent_channels: int
    verbosity: int

    def __init__(self, verbosity):
        self.scent_weights = np.array([
            self.BEANS_SCENT_WEIGHT,
            self.RANCID_BEANS_SCENT_WEIGHT,
            self.FRUIT_SCENT_WEIGHT,
            self.RANCID_FRUIT_SCENT_WEIGHT
        ])
        self.n_scent_channels = self.scent_weights.shape[1]
        self.verbosity = verbosity

    def generate(self, food_items, obstacle_mask):
        size = obstacle_mask._shape_xy[0]
        food_scents = np.zeros((size, size, self.n_scent_channels, len(food_items)), dtype=np.float)
        # for k, (i, j, food_item) in enumerate(food_items):
        #     for _i, _j in product(range(size), range(size)):
        #         d = np.sqrt(self._dist(i, j, _i, _j))
        #         d = max(1., d)
        #         food_scents[_i, _j, :, k] = self.scent_weights[food_item] / d**1.5
        #
        # # zeroes scent for obstacle cells
        # food_scents *= ~obstacle_mask.reshape(obstacle_mask.shape + (1, 1,))

        return food_scents, self.n_scent_channels

    @staticmethod
    def _dist(i, j, _i, _j):
        return (i - _i) ** 2 + (j - _j) ** 2

class FoodGenerator:
    BEANS_DISTRIBUTION = [.3, .1, .25, .15, .1]
    RANCID_BEANS_DISTRIBUTION = [.25, .15, .1, .35, .15]
    FRUIT_DISTRIBUTION = [.15, .35, .2, .1, .2]
    RANCID_FRUIT_DISTRIBUTION = [.15, .3, .15, .15, .25]
    FOOD_TYPES_DISTRIBUTION = [.37, .33, .17, .13]

    verbosity: int

    def __init__(self, verbosity):
        self.food_distribution = np.array([
            self.BEANS_DISTRIBUTION,
            self.RANCID_BEANS_DISTRIBUTION,
            self.FRUIT_DISTRIBUTION,
            self.RANCID_FRUIT_DISTRIBUTION
        ])
        self.food_types_distribution = np.array(self.FOOD_TYPES_DISTRIBUTION)
        self.n_food_types = len(self.FOOD_TYPES_DISTRIBUTION)
        self.verbosity = verbosity

    def generate(self, areas_map, obstacle_mask, seed):
        rnd = np.random.default_rng(seed=seed)
        size = areas_map._shape_xy[0]
        n_foods = max(int((size - 2) ** 1.), self.n_food_types)

        foods = rnd.choice(
            self.n_food_types,
            p=self.food_types_distribution,
            size=n_foods-self.n_food_types
        )
        foods = list(range(self.n_food_types)) + list(foods)
        trace(self.verbosity, 2, f'Food: {np.unique(foods, return_counts=True)[1]}')

        food_mask = np.zeros_like(areas_map, dtype=np.bool)
        food_items = []
        food_map = np.full_like(areas_map, -1, dtype=np.int8)
        food_probs = np.empty_like(areas_map, dtype=np.float)
        for food_item in foods:
            food_probs.fill(0.)
            for i,j in product(range(size), range(size)):
                if obstacle_mask[i, j] or food_mask[i, j]:
                    continue
                food_probs[i, j] = self.food_distribution[food_item, areas_map[i, j]]

            food_probs /= food_probs.sum()
            ind = rnd.choice(size*size, p=food_probs.flatten())
            i, j = divmod(ind, size)
            food_items.append((i, j, food_item))
            food_mask[i, j] = True
            food_map[i, j] = food_item
        return food_items, food_map, food_mask, self.n_food_types


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

class AreasGenerator:
    size: int
    n_area_types: int
    verbosity: int

    def __init__(self, size, n_area_types, verbosity):
        self.size = size
        self.n_area_types = n_area_types
        self.verbosity = verbosity

    def generate(self, seed):
        rnd = np.random.default_rng(seed=seed)
        area_centers, area_types = self._spawn_area_centers(rnd)
        trace(self.verbosity, 3, area_types)
        # self._plot_area_centers(area_centers, area_types)
        areas_map = self._spawn_areas(area_centers, area_types)

        trace_image(self.verbosity, 3, areas_map)
        return areas_map, self.n_area_types

    def _spawn_areas(self, area_centers, area_types):
        size = self.size
        n_centers = len(area_types)

        areas_map = np.empty((size, size), dtype=np.int8)
        for i, j in product(range(size), range(size)):
            distances = [
                self._dist(i, j, area_centers[0][k], area_centers[1][k])
                for k in range(n_centers)
            ]
            areas_map[i, j] = area_types[np.argmin(distances)]
        return areas_map

    @staticmethod
    def _dist(i, j, _i, _j):
        return (i - _i)**2 + (j - _j)**2

    def _spawn_area_centers(self, rnd):
        size = self.size
        n_areas = rnd.integers(
            max(size//3, self.n_area_types),
            max(int(1.7 * self.size), self.n_area_types)
        )
        trace(self.verbosity, 2, f'{n_areas} areas of {self.n_area_types} types are spawned')

        flatten_centers = rnd.integers(size * size, size=n_areas)
        area_centers = np.divmod(flatten_centers, size)
        area_types = rnd.integers(self.n_area_types, size=n_areas)
        return area_centers, area_types

    def _plot_area_centers(self, area_centers, area_types):
        size = self.size
        area = np.zeros((size, size), dtype=np.int8)
        for i, t in enumerate(area_types):
            area[area_centers[0][i], area_centers[1][i]] = t + 1

        plt.imshow(area)
        plt.show()


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


def trace_image(verbosity, req_verbosity, img):
    if verbosity >= req_verbosity:
        plt.imshow(img)
        plt.show()