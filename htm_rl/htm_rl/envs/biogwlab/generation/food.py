import numpy as np

from htm_rl.envs.biogwlab.environment_state import EnvironmentState


class FoodGenerator:
    def add(self, state: EnvironmentState):
        rnd = np.random.default_rng(seed=state.seed)
        n_cells = state.n_cells

        # n_foods = max(int(n_cells ** .5 - 2), 1)
        # print(n_foods)

        n_foods = 1

        # work in flatten then reshape
        empty_positions = np.flatnonzero(~state.obstacle_mask)
        food_positions = rnd.choice(empty_positions, size=n_foods, replace=False)

        food_mask = np.zeros(n_cells, dtype=np.bool)
        food_mask[food_positions] = True
        food_mask = food_mask.reshape(state.shape)

        state.food_mask = food_mask
        state.n_foods = n_foods
