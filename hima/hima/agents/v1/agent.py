import numpy as np

from hima.agents.agent import Agent
from hima.common.sdr import SparseSdr


class RndAgent(Agent):
    n_actions: int

    def __init__(
            self,
            n_actions: int,
            seed: int,
    ):
        self.n_actions = n_actions
        self.rng = np.random.default_rng(seed)

    @property
    def name(self):
        return 'rnd'

    def act(self, reward: float, state: SparseSdr, first: bool):
        return self.rng.integers(self.n_actions)

    def reset(self):
        pass

class ExploreAgent:
    def __init__(self, grid_size: int):
        self.map = np.zeros((grid_size, grid_size))
        self.cell_size = 40. / grid_size
        self.centers = np.linspace(self.cell_size / 2, 40 - self.cell_size / 2, grid_size)
        self.actions = {
            'L': 1,
            'R': 2,
            'F': 3,
            'FL': 4,
            'FR': 5,
            'B': 6
        }
        self.prev_pos = None
        self.cur_goal = None
        self.data = {(i, j): [] for i in range(grid_size) for j in range(grid_size)}

    def act(self, position: tuple[float], velocity: tuple[float], camera: np.ndarray):
        i = np.argmin(np.abs(position[0] - self.centers))
        j = np.argmin(np.abs(position[2] - self.centers))
        self.data[(i, j)].append(camera)
        self.map[i, j] += 1
        pos = np.array([position[0], position[2]])

        if self.prev_pos is None:
            self.prev_pos = pos

        vel = pos - self.prev_pos
        self.prev_pos = pos
        if np.sqrt(vel @ vel.T) < 1e-2:
            return self.actions['FL']

        if self.cur_goal is None:
            self.cur_goal = np.random.randint(0, len(self.centers), 2)
        elif i == self.cur_goal[0] and j == self.cur_goal[1]:
            p = self.map.max() - self.map.flatten()
            p = p / p.sum()
            ind = np.random.choice(np.arange(len(p)), p=p)
            self.cur_goal = np.unravel_index(ind, self.map.shape)

        g_i, g_j = self.cur_goal
        vec2goal = np.array([self.centers[g_i] - position[0], self.centers[g_j] - position[2]])
        prod = vel[0] * vec2goal[1] - vel[1] * vec2goal[0]

        if np.abs(prod) < 1:
            return self.actions['F']
        if prod < 0:
            return self.actions['FR']
        else:
            return self.actions['FL']
