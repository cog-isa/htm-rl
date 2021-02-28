import numpy as np


class ObstacleGenerator:
    pass


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
